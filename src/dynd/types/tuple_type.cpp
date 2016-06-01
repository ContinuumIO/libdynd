//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/buffer.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/type_type.hpp>

using namespace std;
using namespace dynd;

void ndt::tuple_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  const uintptr_t *data_offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
  o << "[";
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    get_field_type(i).print_data(o, arrmeta + m_arrmeta_offsets[i], data + data_offsets[i]);
    if (i != i_end - 1) {
      o << ", ";
    }
  }
  o << "]";
}

bool ndt::tuple_type::is_expression() const {
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    if (get_field_type(i).is_expression()) {
      return true;
    }
  }
  return false;
}

bool ndt::tuple_type::is_unique_data_owner(const char *arrmeta) const {
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &ft = get_field_type(i);
    if (!ft.is_builtin() && !ft.extended()->is_unique_data_owner(arrmeta + m_arrmeta_offsets[i])) {
      return false;
    }
  }
  return true;
}

size_t ndt::tuple_type::get_default_data_size() const {
  intptr_t field_count = get_field_count();
  // Default layout is to match the field order - could reorder the elements for
  // more efficient packing
  size_t s = 0;
  for (intptr_t i = 0; i != field_count; ++i) {
    const type &ft = get_field_type(i);
    s = inc_to_alignment(s, ft.get_data_alignment());
    if (!ft.is_builtin()) {
      s += ft.extended()->get_default_data_size();
    } else {
      s += ft.get_data_size();
    }
  }
  s = inc_to_alignment(s, this->m_data_alignment);
  return s;
}

void ndt::tuple_type::get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta,
                                const char *DYND_UNUSED(data)) const {
  out_shape[i] = m_variadic ? -1 : get_field_count();
  if (i < ndim - 1) {
    dimvector tmpshape(ndim);
    // Accumulate the shape from all the field shapes
    for (intptr_t fi = 0, fi_end = get_field_count(); fi != fi_end; ++fi) {
      const type &ft = get_field_type(fi);
      if (!ft.is_builtin()) {
        ft.extended()->get_shape(ndim, i + 1, tmpshape.get(), arrmeta ? (arrmeta + m_arrmeta_offsets[fi]) : NULL, NULL);
      } else {
        stringstream ss;
        ss << "requested too many dimensions from type " << ft;
        throw runtime_error(ss.str());
      }
      if (fi == 0) {
        // Copy the shape from the first field
        memcpy(out_shape + i + 1, tmpshape.get() + i + 1, (ndim - i - 1) * sizeof(intptr_t));
      } else {
        // Merge the shape from the rest
        for (intptr_t k = i + 1; k < ndim; ++k) {
          // If we see different sizes, make the output -1
          if (out_shape[k] != -1 && out_shape[k] != tmpshape[k]) {
            out_shape[k] = -1;
          }
        }
      }
    }
  }
}

void ndt::tuple_type::get_vars(std::unordered_set<std::string> &vars) const {
  for (intptr_t i = 0; i < m_field_count; ++i) {
    get_field_type(i).get_vars(vars);
  }
}

ndt::type ndt::tuple_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                              const type &root_tp, bool leading_dimension) const {
  if (nindices == 0) {
    return type(this, true);
  } else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, get_field_count(), current_i, &root_tp, remove_dimension, start_index,
                              index_stride, dimension_size);
    if (remove_dimension) {
      return get_field_type(start_index)
          .apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, leading_dimension);
    } else if (nindices == 1 && start_index == 0 && index_stride == 1 && dimension_size == get_field_count()) {
      // This is a do-nothing index, keep the same type
      return type(this, true);
    } else {
      // Take the subset of the fields in-place
      std::vector<ndt::type> tmp_field_types(dimension_size);

      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        tmp_field_types[i] =
            get_field_type(idx).apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false);
      }

      return make_type<tuple_type>(tmp_field_types.size(), tmp_field_types.data());
    }
  }
}

intptr_t ndt::tuple_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                             const type &result_tp, char *out_arrmeta,
                                             const nd::memory_block &embedded_reference, size_t current_i,
                                             const type &root_tp, bool leading_dimension, char **inout_data,
                                             nd::memory_block &inout_dataref) const {
  if (nindices == 0) {
    // If there are no more indices, copy the arrmeta verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  } else {
    const uintptr_t *offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, get_field_count(), current_i, &root_tp, remove_dimension, start_index,
                              index_stride, dimension_size);
    if (remove_dimension) {
      nd::memory_block tmp;
      const type &ft = get_field_type(start_index);
      intptr_t offset = offsets[start_index];
      if (!ft.is_builtin()) {
        if (leading_dimension) {
          // In the case of a leading dimension, first bake the offset into
          // the data pointer, so that it's pointing at the right element
          // for the collapsing of leading dimensions to work correctly.
          *inout_data += offset;
          offset = ft.extended()->apply_linear_index(
              nindices - 1, indices + 1, arrmeta + m_arrmeta_offsets[start_index], result_tp, out_arrmeta,
              embedded_reference, current_i + 1, root_tp, true, inout_data, inout_dataref);
        } else {
          offset += ft.extended()->apply_linear_index(nindices - 1, indices + 1,
                                                      arrmeta + m_arrmeta_offsets[start_index], result_tp, out_arrmeta,
                                                      embedded_reference, current_i + 1, root_tp, false, NULL, tmp);
        }
      }
      return offset;
    } else {
      nd::memory_block tmp;
      intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_arrmeta);
      const tuple_type *result_e_dt = result_tp.extended<tuple_type>();
      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        out_offsets[i] = offsets[idx];
        const type &ft = result_e_dt->get_field_type(i);
        if (!ft.is_builtin()) {
          out_offsets[i] +=
              ft.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + m_arrmeta_offsets[idx], ft,
                                                out_arrmeta + result_e_dt->get_arrmeta_offset(i), embedded_reference,
                                                current_i + 1, root_tp, false, NULL, tmp);
        }
      }
      return 0;
    }
  }
}

void ndt::tuple_type::print_type(std::ostream &o) const {
  // Use the tuple datashape syntax
  o << "(";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    if (i != 0) {
      o << ", ";
    }
    o << get_field_type(i);
  }
  if (m_variadic) {
    o << ", ...)";
  } else {
    o << ")";
  }
}

void ndt::tuple_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                            type &out_transformed_tp, bool &out_was_transformed) const {
  std::vector<ndt::type> tmp_field_types(m_field_count);
  bool was_transformed = false;

  for (intptr_t i = 0; i < m_field_count; ++i) {
    tmp_field_types[i] = make_type<type_type>();
  }

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    transform_fn(get_field_type(i), arrmeta_offset + get_arrmeta_offset(i), extra, tmp_field_types[i], was_transformed);
  }
  if (was_transformed) {
    out_transformed_tp = make_type<tuple_type>(tmp_field_types.size(), tmp_field_types.data(), m_variadic);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::tuple_type::get_canonical_type() const {
  std::vector<ndt::type> tmp_field_types(m_field_count);

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    tmp_field_types[i] = get_field_type(i).get_canonical_type();
  }

  return make_type<tuple_type>(tmp_field_types.size(), tmp_field_types.data(), m_variadic);
}

bool ndt::tuple_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_id() == tuple_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::tuple_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != tuple_id) {
    return false;
  } else {
    const tuple_type *dt = static_cast<const tuple_type *>(&rhs);
    return get_data_alignment() == dt->get_data_alignment() && m_field_types == dt->m_field_types &&
           m_variadic == dt->m_variadic;
  }
}

void ndt::tuple_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const {
  uintptr_t *data_offsets = reinterpret_cast<uintptr_t *>(arrmeta);
  const vector<type> &field_tps = get_field_types();
  // If the arrmeta has data offsets, fill them in
  if (data_offsets != NULL) {
    fill_default_data_offsets(get_field_count(), field_tps.data(), data_offsets);
  }

  // Default construct the arrmeta for all the fields
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &tp = field_tps[i];
    if (!tp.is_builtin()) {
      try {
        tp.extended()->arrmeta_default_construct(arrmeta + m_arrmeta_offsets[i], blockref_alloc);
      } catch (...) {
        // Since we're explicitly controlling the memory, need to manually do
        // the cleanup too
        for (intptr_t j = 0; j < i; ++j) {
          const type &ft = get_field_type(j);
          if (!ft.is_builtin()) {
            ft.extended()->arrmeta_destruct(arrmeta + m_arrmeta_offsets[i]);
          }
        }
        throw;
      }
    }
  }
}

void ndt::tuple_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                             const nd::memory_block &embedded_reference) const {
  uintptr_t *dst_data_offsets = reinterpret_cast<uintptr_t *>(dst_arrmeta);
  if (dst_data_offsets != 0) {
    // Copy all the field offsets
    memcpy(dst_data_offsets, reinterpret_cast<const uintptr_t *>(src_arrmeta), get_field_count() * sizeof(uintptr_t));
  }
  // Copy construct all the field's arrmeta
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin()) {
      field_dt.extended()->arrmeta_copy_construct(dst_arrmeta + m_arrmeta_offsets[i],
                                                  src_arrmeta + m_arrmeta_offsets[i], embedded_reference);
    }
  }
}

void ndt::tuple_type::arrmeta_reset_buffers(char *arrmeta) const {
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (field_dt.get_arrmeta_size() > 0) {
      field_dt.extended()->arrmeta_reset_buffers(arrmeta + m_arrmeta_offsets[i]);
    }
  }
}

void ndt::tuple_type::arrmeta_finalize_buffers(char *arrmeta) const {
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin()) {
      field_dt.extended()->arrmeta_finalize_buffers(arrmeta + m_arrmeta_offsets[i]);
    }
  }
}

void ndt::tuple_type::arrmeta_destruct(char *arrmeta) const {
  for (intptr_t i = 0, i_end = get_field_count(); i != i_end; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin()) {
      field_dt.extended()->arrmeta_destruct(arrmeta + m_arrmeta_offsets[i]);
    }
  }
}

void ndt::tuple_type::data_destruct(const char *arrmeta, char *data) const {
  const uintptr_t *data_offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
  intptr_t field_count = get_field_count();
  for (intptr_t i = 0; i != field_count; ++i) {
    const type &ft = get_field_type(i);
    if (ft.get_flags() & type_flag_destructor) {
      ft.extended()->data_destruct(arrmeta + m_arrmeta_offsets[i], data + data_offsets[i]);
    }
  }
}

void ndt::tuple_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const {
  const uintptr_t *data_offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
  intptr_t field_count = get_field_count();
  // Destruct all the fields a chunk at a time, in an
  // attempt to have some kind of locality
  while (count > 0) {
    size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
    for (intptr_t i = 0; i != field_count; ++i) {
      const type &ft = get_field_type(i);
      if (ft.get_flags() & type_flag_destructor) {
        ft.extended()->data_destruct_strided(arrmeta + m_arrmeta_offsets[i], data + data_offsets[i], stride,
                                             chunk_size);
      }
    }
    data += stride * chunk_size;
    count -= chunk_size;
  }
}

void ndt::tuple_type::foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback,
                                      void *callback_data) const {
  if (get_field_count() != 0) {
    const uintptr_t *data_offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
    for (intptr_t i = 0, i_end = get_field_count(); i < i_end; ++i) {
      callback(get_field_type(i), arrmeta + m_arrmeta_offsets[i], data + data_offsets[i], callback_data);
    }
  }
}

void ndt::tuple_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const {
  const size_t *data_offsets = reinterpret_cast<const size_t *>(arrmeta);
  o << indent << "tuple arrmeta\n";
  o << indent << " field offsets: ";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    o << data_offsets[i];
    if (i != i_end - 1) {
      o << ", ";
    }
  }
  o << "\n";
  for (intptr_t i = 0; i < m_field_count; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin() && field_dt.extended()->get_arrmeta_size() > 0) {
      o << indent << " field " << i << " arrmeta:\n";
      field_dt.extended()->arrmeta_debug_print(arrmeta + m_arrmeta_offsets[i], o, indent + "  ");
    }
  }
}

bool ndt::tuple_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  intptr_t candidate_field_count = candidate_tp.extended<tuple_type>()->get_field_count();
  bool candidate_variadic = candidate_tp.extended<tuple_type>()->is_variadic();

  if ((m_field_count == candidate_field_count && !candidate_variadic) ||
      ((candidate_field_count >= m_field_count) && m_variadic)) {
    // Match against the types
    const std::vector<type> &candidate_fields = candidate_tp.extended<tuple_type>()->get_field_types();
    for (intptr_t i = 0; i != m_field_count; ++i) {
      if (!m_field_types[i].match(candidate_fields[i], tp_vars)) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::tuple_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["field_types"] = {ndt::type_for(m_field_types), reinterpret_cast<const char *>(&m_field_types)};
  properties["metadata_offsets"] = {ndt::type_for(m_arrmeta_offsets),
                                    reinterpret_cast<const char *>(&m_arrmeta_offsets)};

  return properties;
}
