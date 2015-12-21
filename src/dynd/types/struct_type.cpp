//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/base_property_kernel.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>
#include <dynd/func/assignment.hpp>

using namespace std;
using namespace dynd;

ndt::struct_type::struct_type(const nd::array &field_names, const nd::array &field_types, bool variadic)
    : tuple_type(struct_type_id, field_types, type_flag_none, true, variadic), m_field_names(field_names)
{
  /*
    if (!nd::ensure_immutable_contig<std::string>(m_field_names)) {
      stringstream ss;
      ss << "dynd struct field names requires an array of strings, got an "
            "array with type " << m_field_names.get_type();
      throw invalid_argument(ss.str());
    }
  */

  // Make sure that the number of names matches
  intptr_t name_count = reinterpret_cast<const fixed_dim_type_arrmeta *>(m_field_names.get()->metadata())->dim_size;
  if (name_count != m_field_count) {
    stringstream ss;
    ss << "dynd struct type requires that the number of names, " << name_count << " matches the number of types, "
       << m_field_count;
    throw invalid_argument(ss.str());
  }

  this->kind = variadic ? kind_kind : struct_kind;

  create_array_properties();
}

ndt::struct_type::~struct_type() {}

intptr_t ndt::struct_type::get_field_index(const char *field_name_begin, const char *field_name_end) const
{
  size_t size = field_name_end - field_name_begin;
  if (size > 0) {
    char firstchar = *field_name_begin;
    intptr_t field_count = get_field_count();
    const char *fn_ptr = m_field_names.cdata();
    intptr_t fn_stride = reinterpret_cast<const fixed_dim_type_arrmeta *>(m_field_names.get()->metadata())->stride;
    for (intptr_t i = 0; i != field_count; ++i, fn_ptr += fn_stride) {
      const string *fn = reinterpret_cast<const string *>(fn_ptr);
      const char *begin = fn->begin(), *end = fn->end();
      if ((size_t)(end - begin) == size && *begin == firstchar) {
        if (memcmp(fn->begin(), field_name_begin, size) == 0) {
          return i;
        }
      }
    }
  }

  return -1;
}

static bool is_simple_identifier_name(const char *begin, const char *end)
{
  if (begin == end) {
    return false;
  }
  else {
    char c = *begin++;
    if (!(('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_')) {
      return false;
    }
    while (begin < end) {
      c = *begin++;
      if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_')) {
        return false;
      }
    }
    return true;
  }
}

void ndt::struct_type::print_type(std::ostream &o) const
{
  // Use the record datashape syntax
  o << "{";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    if (i != 0) {
      o << ", ";
    }
    const string &fn = get_field_name_raw(i);
    if (is_simple_identifier_name(fn.begin(), fn.end())) {
      o.write(fn.begin(), fn.end() - fn.begin());
    }
    else {
      print_escaped_utf8_string(o, fn.begin(), fn.end(), true);
    }
    o << ": " << get_field_type(i);
  }
  if (m_variadic) {
    o << ", ...}";
  }
  else {
    o << "}";
  }
}

void ndt::struct_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                             type &out_transformed_tp, bool &out_was_transformed) const
{
  nd::array tmp_field_types(nd::empty(m_field_count, make_type<type_type>()));
  type *tmp_field_types_raw = reinterpret_cast<type *>(tmp_field_types.data());

  bool was_transformed = false;
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    transform_fn(get_field_type(i), arrmeta_offset + get_arrmeta_offset(i), extra, tmp_field_types_raw[i],
                 was_transformed);
  }
  if (was_transformed) {
    tmp_field_types.flag_as_immutable();
    out_transformed_tp = struct_type::make(m_field_names, tmp_field_types, m_variadic);
    out_was_transformed = true;
  }
  else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::struct_type::get_canonical_type() const
{
  nd::array tmp_field_types(nd::empty(m_field_count, make_type<type_type>()));
  type *tmp_field_types_raw = reinterpret_cast<type *>(tmp_field_types.data());

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    tmp_field_types_raw[i] = get_field_type(i).get_canonical_type();
  }

  tmp_field_types.flag_as_immutable();
  return struct_type::make(m_field_names, tmp_field_types, m_variadic);
}

ndt::type ndt::struct_type::at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const
{
  // Bounds-checking of the index
  i0 = apply_single_index(i0, m_field_count, NULL);
  if (inout_arrmeta) {
    char *arrmeta = const_cast<char *>(*inout_arrmeta);
    // Modify the arrmeta
    *inout_arrmeta += get_arrmeta_offsets_raw()[i0];
    // If requested, modify the data
    if (inout_data) {
      *inout_data += get_arrmeta_data_offsets(arrmeta)[i0];
    }
  }
  return get_field_type(i0);
}

bool ndt::struct_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else if (src_tp.get_type_id() == struct_type_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

size_t ndt::struct_type::make_comparison_kernel(void *ckb, intptr_t ckb_offset, const type &src0_dt,
                                                const char *src0_arrmeta, const type &src1_dt, const char *src1_arrmeta,
                                                comparison_type_t comptype, const eval::eval_context *ectx) const
{
  if (this == src0_dt.extended()) {
    if (*this == *src1_dt.extended()) {
      return make_tuple_comparison_kernel(ckb, ckb_offset, src0_dt, src0_arrmeta, src1_arrmeta, comptype, ectx);
    }
    else if (src1_dt.get_kind() == struct_kind) {
      // TODO
    }
  }

  throw not_comparable_error(src0_dt, src1_dt, comptype);
}

bool ndt::struct_type::operator==(const base_type &rhs) const
{
  if (this == &rhs) {
    return true;
  }
  else if (rhs.get_type_id() != struct_type_id) {
    return false;
  }
  else {
    const struct_type *dt = static_cast<const struct_type *>(&rhs);
    return get_data_alignment() == dt->get_data_alignment() && m_field_types.equals_exact(dt->m_field_types) &&
           m_field_names.equals_exact(dt->m_field_names) && m_variadic == dt->m_variadic;
  }
}

void ndt::struct_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const
{
  const size_t *offsets = reinterpret_cast<const size_t *>(arrmeta);
  o << indent << "struct arrmeta\n";
  o << indent << " field offsets: ";
  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    o << offsets[i];
    if (i != i_end - 1) {
      o << ", ";
    }
  }
  o << "\n";
  const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
  for (intptr_t i = 0; i < m_field_count; ++i) {
    const type &field_dt = get_field_type(i);
    if (!field_dt.is_builtin() && field_dt.extended()->get_arrmeta_size() > 0) {
      o << indent << " field " << i << " (name ";
      const string &fnr = get_field_name_raw(i);
      o.write(fnr.begin(), fnr.end() - fnr.begin());
      o << ") arrmeta:\n";
      field_dt.extended()->arrmeta_debug_print(arrmeta + arrmeta_offsets[i], o, indent + "  ");
    }
  }
}

ndt::type ndt::struct_type::apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i,
                                               const type &root_tp, bool leading_dimension) const
{
  if (nindices == 0) {
    return type(this, true);
  }
  else {
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_field_count, current_i, &root_tp, remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      return get_field_type(start_index)
          .apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, leading_dimension);
    }
    else if (nindices == 1 && start_index == 0 && index_stride == 1 && dimension_size == m_field_count) {
      // This is a do-nothing index, keep the same type
      return type(this, true);
    }
    else {
      // Take the subset of the fields in-place
      nd::array tmp_field_types(nd::empty(dimension_size, make_type<type_type>()));
      type *tmp_field_types_raw = reinterpret_cast<type *>(tmp_field_types.data());

      // Make an "N * string" array without copying the actual
      // string text data. TODO: encapsulate this into a function.
      string *string_arr_ptr;
      type stp = ndt::make_type<ndt::string_type>();
      type tp = make_fixed_dim(dimension_size, stp);
      nd::array tmp_field_names = nd::empty(tp);
      string_arr_ptr = reinterpret_cast<string *>(tmp_field_names.data());

      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        tmp_field_types_raw[i] =
            get_field_type(idx).apply_linear_index(nindices - 1, indices + 1, current_i + 1, root_tp, false);
        string_arr_ptr[i] = get_field_name_raw(idx);
      }

      tmp_field_types.flag_as_immutable();
      return struct_type::make(tmp_field_names, tmp_field_types);
    }
  }
}

intptr_t ndt::struct_type::apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta,
                                              const type &result_tp, char *out_arrmeta,
                                              const intrusive_ptr<memory_block_data> &embedded_reference,
                                              size_t current_i, const type &root_tp, bool leading_dimension,
                                              char **inout_data, intrusive_ptr<memory_block_data> &inout_dataref) const
{
  if (nindices == 0) {
    // If there are no more indices, copy the arrmeta verbatim
    arrmeta_copy_construct(out_arrmeta, arrmeta, embedded_reference);
    return 0;
  }
  else {
    const uintptr_t *offsets = get_data_offsets(arrmeta);
    const uintptr_t *arrmeta_offsets = get_arrmeta_offsets_raw();
    bool remove_dimension;
    intptr_t start_index, index_stride, dimension_size;
    apply_single_linear_index(*indices, m_field_count, current_i, &root_tp, remove_dimension, start_index, index_stride,
                              dimension_size);
    if (remove_dimension) {
      const type &dt = get_field_type(start_index);
      intptr_t offset = offsets[start_index];
      if (!dt.is_builtin()) {
        if (leading_dimension) {
          // In the case of a leading dimension, first bake the offset into
          // the data pointer, so that it's pointing at the right element
          // for the collapsing of leading dimensions to work correctly.
          *inout_data += offset;
          offset = dt.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + arrmeta_offsets[start_index],
                                                     result_tp, out_arrmeta, embedded_reference, current_i + 1, root_tp,
                                                     true, inout_data, inout_dataref);
        }
        else {
          intrusive_ptr<memory_block_data> tmp;
          offset += dt.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + arrmeta_offsets[start_index],
                                                      result_tp, out_arrmeta, embedded_reference, current_i + 1,
                                                      root_tp, false, NULL, tmp);
        }
      }
      return offset;
    }
    else {
      intrusive_ptr<memory_block_data> tmp;
      intptr_t *out_offsets = reinterpret_cast<intptr_t *>(out_arrmeta);
      const struct_type *result_e_dt = result_tp.extended<struct_type>();
      for (intptr_t i = 0; i < dimension_size; ++i) {
        intptr_t idx = start_index + i * index_stride;
        out_offsets[i] = offsets[idx];
        const type &dt = result_e_dt->get_field_type(i);
        if (!dt.is_builtin()) {
          out_offsets[i] +=
              dt.extended()->apply_linear_index(nindices - 1, indices + 1, arrmeta + arrmeta_offsets[idx], dt,
                                                out_arrmeta + result_e_dt->get_arrmeta_offset(i), embedded_reference,
                                                current_i + 1, root_tp, false, NULL, tmp);
        }
      }
      return 0;
    }
  }
}

/*
static nd::array property_get_field_names(const ndt::type &tp)
{
  return tp.extended<ndt::struct_type>()->get_field_names();
}

static nd::array property_get_field_types(const ndt::type &tp)
{
  return tp.extended<ndt::struct_type>()->get_field_types();
}

static nd::array property_get_arrmeta_offsets(const ndt::type &tp)
{
  return tp.extended<ndt::struct_type>()->get_arrmeta_offsets();
}
*/

std::map<std::string, nd::callable> ndt::struct_type::get_dynamic_type_properties() const
{
  struct field_types_kernel : nd::base_property_kernel<field_types_kernel> {
    field_types_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<field_types_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_assign(dst_tp, dst_arrmeta, dst, dst_tp, tp.extended<struct_type>()->m_field_types.get()->metadata(),
                        tp.extended<struct_type>()->m_field_types.cdata());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = tp.extended<struct_type>()->m_field_types.get_type();
    }
  };

  struct field_names_kernel : nd::base_property_kernel<field_names_kernel> {
    field_names_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<field_names_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_assign(dst_tp, dst_arrmeta, dst, dst_tp, tp.extended<struct_type>()->m_field_names.get()->metadata(),
                        tp.extended<struct_type>()->m_field_names.cdata());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = tp.extended<struct_type>()->m_field_names.get_type();
    }
  };

  struct arrmeta_offsets_kernel : nd::base_property_kernel<arrmeta_offsets_kernel> {
    arrmeta_offsets_kernel(const ndt::type &tp, const ndt::type &dst_tp, const char *dst_arrmeta)
        : base_property_kernel<arrmeta_offsets_kernel>(tp, dst_tp, dst_arrmeta)
    {
    }

    void single(char *dst, char *const *DYND_UNUSED(src))
    {
      typed_data_assign(dst_tp, dst_arrmeta, dst, dst_tp,
                        tp.extended<struct_type>()->m_arrmeta_offsets.get()->metadata(),
                        tp.extended<struct_type>()->m_arrmeta_offsets.cdata());
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 const dynd::nd::array &DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      const type &tp = *reinterpret_cast<const ndt::type *>(data);
      dst_tp = tp.extended<struct_type>()->m_arrmeta_offsets.get_type();
    }
  };

  std::map<std::string, nd::callable> properties;
  properties["field_types"] = nd::callable::make<field_types_kernel>(type("(self: type) -> Any"));
  properties["field_names"] = nd::callable::make<field_names_kernel>(type("(self: type) -> Any"));
  properties["arrmeta_offsets"] = nd::callable::make<field_names_kernel>(type("(self: type) -> Any"));

  return properties;
}

namespace dynd {
namespace nd {

  struct get_array_field_kernel : nd::base_kernel<get_array_field_kernel> {
    static const kernel_request_t kernreq = kernel_request_call;

    array self;
    intptr_t i;

    get_array_field_kernel(const array &self, intptr_t i) : self(self), i(i) {}

    void call(array *dst, array *const *DYND_UNUSED(src))
    {
      array res = helper(self, i);
      *dst = res;
    }

    static void resolve_dst_type(char *static_data, char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0], *reinterpret_cast<intptr_t *>(static_data)).get_type();
    }

    static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      get_array_field_kernel::make(ckb, kernreq, ckb_offset, kwds[0], *reinterpret_cast<intptr_t *>(static_data));
      return ckb_offset;
    }

    static array helper(const array &n, intptr_t i)
    {
      // Get the nd::array 'self' parameter
      intptr_t undim = n.get_ndim();
      ndt::type udt = n.get_dtype();
      if (udt.get_kind() == expr_kind) {
        std::string field_name = udt.value_type().extended<ndt::struct_type>()->get_field_name(i);
        return n.replace_dtype(ndt::make_type<ndt::adapt_type>(
            udt.value_type().extended<ndt::struct_type>()->get_field_type(i), udt, nd::callable(), nd::callable()));
      }
      else {
        if (undim == 0) {
          return n(i);
        }
        else {
          shortvector<irange> idx(undim + 1);
          idx[undim] = irange(i);
          return n.at_array(undim + 1, idx.get());
        }
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd

ndt::struct_type::struct_type(struct_type *other)
    : tuple_type(struct_type_id, {ndt::type(other, false)}, type_flag_none, false, false)
{
  // Equivalent to ndt::struct_type::make(ndt::make_ndarrayarg(), "self");
  // but hardcoded to break the dependency of struct_type::array_parameters_type
  uintptr_t metaoff[1] = {0};
  m_arrmeta_offsets = nd::array(metaoff);
  // The data offsets also consist of one zero
  //    m_data_offsets = m_arrmeta_offsets;
  // Inherit any operand flags from the fields
  this->flags |= (ndt::any_kind_type::make().get_flags() & type_flags_operand_inherited);
  this->data_alignment = sizeof(void *);
  this->arrmeta_size = 0;
  this->data_size = sizeof(void *);
  m_field_names = {"self"};
  // Leave m_array_properties so there is no reference loop

  owner = other;
  owner_id = type_type_id;
  owner_use_count = 1;
}

void ndt::struct_type::create_array_properties()
{
  type array_parameters_type(new struct_type(this), true);

  for (intptr_t i = 0, i_end = m_field_count; i != i_end; ++i) {
    nd::callable property = nd::callable::make<nd::get_array_field_kernel>(
        callable_type::make(type("Any"), tuple_type::make(), array_parameters_type), i);
    m_array_properties[get_field_name(i)] = property;

    /*
        property.get()->owner = this;
        property.get()->owner_id = type_type_id;
        property.get()->owner_use_count = 2;
    */
  }
}

std::map<std::string, nd::callable> ndt::struct_type::get_dynamic_array_properties() const
{
  return m_array_properties;
}

bool ndt::struct_type::match(const char *arrmeta, const type &candidate_tp, const char *candidate_arrmeta,
                             std::map<std::string, type> &tp_vars) const
{
  intptr_t candidate_field_count = candidate_tp.extended<struct_type>()->get_field_count();
  bool candidate_variadic = candidate_tp.extended<tuple_type>()->is_variadic();

  if ((m_field_count == candidate_field_count && !candidate_variadic) ||
      ((candidate_field_count >= m_field_count) && m_variadic)) {
    // Compare the field names
    if (m_field_count == candidate_field_count) {
      if (!get_field_names().equals_exact(candidate_tp.extended<struct_type>()->get_field_names())) {
        return false;
      }
    }
    else {
      nd::array leading_field_names = get_field_names();
      if (!leading_field_names.equals_exact(
              candidate_tp.extended<struct_type>()->get_field_names()(irange() < m_field_count))) {
        return false;
      }
    }

    const type *fields = get_field_types_raw();
    const type *candidate_fields = candidate_tp.extended<struct_type>()->get_field_types_raw();
    for (intptr_t i = 0; i < m_field_count; ++i) {
      if (!fields[i].match(arrmeta, candidate_fields[i], candidate_arrmeta, tp_vars)) {
        return false;
      }
    }
    return true;
  }

  return false;
}

nd::array dynd::struct_concat(nd::array lhs, nd::array rhs)
{
  nd::array res;
  if (lhs.is_null()) {
    res = rhs;
    return res;
  }
  if (rhs.is_null()) {
    res = lhs;
    return res;
  }
  const ndt::type &lhs_tp = lhs.get_type(), &rhs_tp = rhs.get_type();
  if (lhs_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "Cannot concatenate array with type " << lhs_tp << " as a struct";
    throw invalid_argument(ss.str());
  }
  if (rhs_tp.get_kind() != struct_kind) {
    stringstream ss;
    ss << "Cannot concatenate array with type " << rhs_tp << " as a struct";
    throw invalid_argument(ss.str());
  }

  // Make an empty shell struct by concatenating the fields together
  intptr_t lhs_n = lhs_tp.extended<ndt::struct_type>()->get_field_count();
  intptr_t rhs_n = rhs_tp.extended<ndt::struct_type>()->get_field_count();
  intptr_t res_n = lhs_n + rhs_n;
  nd::array res_field_names = nd::empty(res_n, ndt::make_type<ndt::string_type>());
  nd::array res_field_types = nd::empty(res_n, ndt::make_type<ndt::type_type>());
  res_field_names(irange(0, lhs_n)).vals() = lhs_tp.extended<ndt::struct_type>()->get_field_names();
  res_field_names(irange(lhs_n, res_n)).vals() = rhs_tp.extended<ndt::struct_type>()->get_field_names();
  res_field_types(irange(0, lhs_n)).vals() = lhs_tp.extended<ndt::struct_type>()->get_field_types();
  res_field_types(irange(lhs_n, res_n)).vals() = rhs_tp.extended<ndt::struct_type>()->get_field_types();
  ndt::type res_tp = ndt::struct_type::make(res_field_names, res_field_types);
  const ndt::type *res_field_tps = res_tp.extended<ndt::struct_type>()->get_field_types_raw();
  res = nd::empty_shell(res_tp);

  // Initialize the default data offsets for the struct arrmeta
  ndt::struct_type::fill_default_data_offsets(res_n, res_tp.extended<ndt::struct_type>()->get_field_types_raw(),
                                              reinterpret_cast<uintptr_t *>(res.get()->metadata()));
  // Get information about the arrmeta layout of the input and res
  const uintptr_t *lhs_arrmeta_offsets = lhs_tp.extended<ndt::struct_type>()->get_arrmeta_offsets_raw();
  const uintptr_t *rhs_arrmeta_offsets = rhs_tp.extended<ndt::struct_type>()->get_arrmeta_offsets_raw();
  const uintptr_t *res_arrmeta_offsets = res_tp.extended<ndt::struct_type>()->get_arrmeta_offsets_raw();
  const char *lhs_arrmeta = lhs.get()->metadata();
  const char *rhs_arrmeta = rhs.get()->metadata();
  char *res_arrmeta = res.get()->metadata();
  // Copy the arrmeta from the input arrays
  for (intptr_t i = 0; i < lhs_n; ++i) {
    const ndt::type &tp = res_field_tps[i];
    if (!tp.is_builtin()) {
      tp.extended()->arrmeta_copy_construct(res_arrmeta + res_arrmeta_offsets[i], lhs_arrmeta + lhs_arrmeta_offsets[i],
                                            lhs.get_data_memblock());
    }
  }
  for (intptr_t i = 0; i < rhs_n; ++i) {
    const ndt::type &tp = res_field_tps[i + lhs_n];
    if (!tp.is_builtin()) {
      tp.extended()->arrmeta_copy_construct(res_arrmeta + res_arrmeta_offsets[i + lhs_n],
                                            rhs_arrmeta + rhs_arrmeta_offsets[i], rhs.get_data_memblock());
    }
  }

  // Get information about the data layout of the input and res
  const uintptr_t *lhs_data_offsets = lhs_tp.extended<ndt::struct_type>()->get_data_offsets(lhs.get()->metadata());
  const uintptr_t *rhs_data_offsets = rhs_tp.extended<ndt::struct_type>()->get_data_offsets(rhs.get()->metadata());
  const uintptr_t *res_data_offsets = res_tp.extended<ndt::struct_type>()->get_data_offsets(res.get()->metadata());
  const char *lhs_data = lhs.cdata();
  const char *rhs_data = rhs.cdata();
  char *res_data = res.data();
  // Copy the data from the input arrays
  for (intptr_t i = 0; i < lhs_n; ++i) {
    const ndt::type &tp = res_field_tps[i];
    typed_data_assign(tp, res_arrmeta + res_arrmeta_offsets[i], res_data + res_data_offsets[i], tp,
                      lhs_arrmeta + lhs_arrmeta_offsets[i], lhs_data + lhs_data_offsets[i]);
  }

  for (intptr_t i = 0; i < rhs_n; ++i) {
    const ndt::type &tp = res_field_tps[i + lhs_n];
    typed_data_assign(tp, res_arrmeta + res_arrmeta_offsets[i + lhs_n], res_data + res_data_offsets[i + lhs_n], tp,
                      rhs_arrmeta + rhs_arrmeta_offsets[i], rhs_data + rhs_data_offsets[i]);
  }

  return res;
}
