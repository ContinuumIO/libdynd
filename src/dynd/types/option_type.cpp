//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/option_type.hpp>

using namespace std;
using namespace dynd;

void dynd::assign_na_builtin(type_id_t value_id, char *data) {
  switch (value_id) {
  // Just use the known value assignments for these builtins
  case bool_id:
    *data = 2;
    return;
  case int8_id:
    *reinterpret_cast<int8_t *>(data) = DYND_INT8_NA;
    return;
  case int16_id:
    *reinterpret_cast<int16_t *>(data) = DYND_INT16_NA;
    return;
  case int32_id:
    *reinterpret_cast<int32_t *>(data) = DYND_INT32_NA;
    return;
  case int64_id:
    *reinterpret_cast<int64_t *>(data) = DYND_INT64_NA;
    return;
  case int128_id:
    *reinterpret_cast<int128 *>(data) = DYND_INT128_NA;
    return;
  case float32_id:
    *reinterpret_cast<uint32_t *>(data) = DYND_FLOAT32_NA_AS_UINT;
    return;
  case float64_id:
    *reinterpret_cast<uint64_t *>(data) = DYND_FLOAT64_NA_AS_UINT;
    return;
  case complex_float32_id:
    reinterpret_cast<uint32_t *>(data)[0] = DYND_FLOAT32_NA_AS_UINT;
    reinterpret_cast<uint32_t *>(data)[1] = DYND_FLOAT32_NA_AS_UINT;
    return;
  case complex_float64_id:
    reinterpret_cast<uint64_t *>(data)[0] = DYND_FLOAT64_NA_AS_UINT;
    reinterpret_cast<uint64_t *>(data)[1] = DYND_FLOAT64_NA_AS_UINT;
    return;
  default:
    break;
  }
}

bool dynd::is_avail_builtin(type_id_t value_id, const char *data) {
  switch (value_id) {
  // Just use the known value assignments for these builtins
  case bool_id:
    return *reinterpret_cast<const unsigned char *>(data) <= 1;
  case int8_id:
    return *reinterpret_cast<const int8_t *>(data) != DYND_INT8_NA;
  case int16_id:
    return *reinterpret_cast<const int16_t *>(data) != DYND_INT16_NA;
  case int32_id:
    return *reinterpret_cast<const int32_t *>(data) != DYND_INT32_NA;
  case uint32_id:
    return *reinterpret_cast<const uint32_t *>(data) != DYND_UINT32_NA;
  case int64_id:
    return *reinterpret_cast<const int64_t *>(data) != DYND_INT64_NA;
  case int128_id:
    return *reinterpret_cast<const int128 *>(data) != DYND_INT128_NA;
  case float32_id:
    return !isnan(*reinterpret_cast<const float *>(data));
  case float64_id:
    return !isnan(*reinterpret_cast<const double *>(data));
  case complex_float32_id:
    return reinterpret_cast<const uint32_t *>(data)[0] != DYND_FLOAT32_NA_AS_UINT ||
           reinterpret_cast<const uint32_t *>(data)[1] != DYND_FLOAT32_NA_AS_UINT;
  case complex_float64_id:
    return reinterpret_cast<const uint64_t *>(data)[0] != DYND_FLOAT64_NA_AS_UINT ||
           reinterpret_cast<const uint64_t *>(data)[1] != DYND_FLOAT64_NA_AS_UINT;
  default:
    return false;
  }
}

void ndt::option_type::get_vars(std::unordered_set<std::string> &vars) const { m_value_tp.get_vars(vars); }

void ndt::option_type::print_type(std::ostream &o) const { o << "?" << m_value_tp; }

void ndt::option_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const {
  if (is_avail_builtin(m_value_tp.get_id(), data)) {
    m_value_tp.print_data(o, arrmeta, data);
  } else {
    o << "NA";
  }
}

bool ndt::option_type::is_expression() const {
  // Even though the pointer is an instance of an base_expr_type,
  // we'll only call it an expression if the target is.
  return m_value_tp.is_expression();
}

bool ndt::option_type::is_unique_data_owner(const char *arrmeta) const {
  if (m_value_tp.get_flags() & type_flag_blockref) {
    return m_value_tp.extended()->is_unique_data_owner(arrmeta);
  }
  return true;
}

void ndt::option_type::transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                                             type &out_transformed_tp, bool &out_was_transformed) const {
  type tmp_tp;
  bool was_transformed = false;
  transform_fn(m_value_tp, arrmeta_offset + 0, extra, tmp_tp, was_transformed);
  if (was_transformed) {
    out_transformed_tp = make_type<option_type>(tmp_tp);
    out_was_transformed = true;
  } else {
    out_transformed_tp = type(this, true);
  }
}

ndt::type ndt::option_type::get_canonical_type() const {
  return make_type<option_type>(m_value_tp.get_canonical_type());
}

ndt::type ndt::option_type::get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim) const {
  if (i == 0) {
    return type(this, true);
  } else {
    return m_value_tp.get_type_at_dimension(inout_arrmeta, i, total_ndim);
  }
}

bool ndt::option_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  if (dst_tp.extended() == this) {
    return ::is_lossless_assignment(m_value_tp, src_tp);
  } else {
    return ::is_lossless_assignment(dst_tp, m_value_tp);
  }
}

bool ndt::option_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != option_id) {
    return false;
  } else {
    const option_type *ot = static_cast<const option_type *>(&rhs);
    return m_value_tp == ot->m_value_tp;
  }
}

void ndt::option_type::arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const {
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_default_construct(arrmeta, blockref_alloc);
  }
}

void ndt::option_type::arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                              const nd::memory_block &embedded_reference) const {
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_copy_construct(dst_arrmeta, src_arrmeta, embedded_reference);
  }
}

void ndt::option_type::arrmeta_reset_buffers(char *arrmeta) const {
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_reset_buffers(arrmeta);
  }
}

void ndt::option_type::arrmeta_finalize_buffers(char *arrmeta) const {
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_finalize_buffers(arrmeta);
  }
}

void ndt::option_type::arrmeta_destruct(char *arrmeta) const {
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_destruct(arrmeta);
  }
}

void ndt::option_type::arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const {
  o << indent << "option arrmeta\n";
  if (!m_value_tp.is_builtin()) {
    m_value_tp.extended()->arrmeta_debug_print(arrmeta, o, indent + " ");
  }
}

void ndt::option_type::data_destruct(const char *arrmeta, char *data) const {
  m_value_tp.extended()->data_destruct(arrmeta, data);
}

void ndt::option_type::data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const {
  m_value_tp.extended()->data_destruct_strided(arrmeta, data, stride, count);
}

bool ndt::option_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  if (candidate_tp.get_id() != option_id) {
    return false;
  }

  return m_value_tp.match(candidate_tp.extended<option_type>()->m_value_tp, tp_vars);
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::option_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["value_type"] = {ndt::type("type"), reinterpret_cast<const char *>(&m_value_tp)};

  return properties;
}
