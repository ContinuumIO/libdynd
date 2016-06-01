//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/typevar_type.hpp>

using namespace std;
using namespace dynd;

void ndt::typevar_type::get_vars(std::unordered_set<std::string> &vars) const { vars.insert(m_name); }

void ndt::typevar_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                   const char *DYND_UNUSED(data)) const {
  throw type_error("Cannot store data of typevar type");
}

void ndt::typevar_type::print_type(std::ostream &o) const {
  // Type variables are barewords starting with a capital letter
  o << m_name;
}

ndt::type ndt::typevar_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                                size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                                bool DYND_UNUSED(leading_dimension)) const {
  throw type_error("Cannot store data of typevar type");
}

intptr_t ndt::typevar_type::apply_linear_index(intptr_t DYND_UNUSED(nindices), const irange *DYND_UNUSED(indices),
                                               const char *DYND_UNUSED(arrmeta), const type &DYND_UNUSED(result_tp),
                                               char *DYND_UNUSED(out_arrmeta),
                                               const nd::memory_block &DYND_UNUSED(embedded_reference),
                                               size_t DYND_UNUSED(current_i), const type &DYND_UNUSED(root_tp),
                                               bool DYND_UNUSED(leading_dimension), char **DYND_UNUSED(inout_data),
                                               nd::memory_block &DYND_UNUSED(inout_dataref)) const {
  throw type_error("Cannot store data of typevar type");
}

bool ndt::typevar_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const {
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    } else if (src_tp.get_id() == typevar_id) {
      return *dst_tp.extended() == *src_tp.extended();
    }
  }

  return false;
}

bool ndt::typevar_type::operator==(const base_type &rhs) const {
  if (this == &rhs) {
    return true;
  } else if (rhs.get_id() != typevar_id) {
    return false;
  } else {
    const typevar_type *tvt = static_cast<const typevar_type *>(&rhs);
    return m_name == tvt->m_name;
  }
}

void ndt::typevar_type::arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {
  throw type_error("Cannot store data of typevar type");
}

void ndt::typevar_type::arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                               const nd::memory_block &DYND_UNUSED(embedded_reference)) const {
  throw type_error("Cannot store data of typevar type");
}

void ndt::typevar_type::arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {
  throw type_error("Cannot store data of typevar type");
}

bool ndt::typevar_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  if (candidate_tp.get_id() == typevar_id) {
    return *this == *candidate_tp.extended();
  }

  if (candidate_tp.get_ndim() > 0 || candidate_tp.get_id() == any_kind_id) {
    return false;
  }

  type &tv_type = tp_vars[m_name];
  if (tv_type.is_null()) {
    // This typevar hasn't been seen yet
    tv_type = candidate_tp;
    return true;
  } else {
    // Make sure the type matches previous
    // instances of the type var
    return candidate_tp == tv_type;
  }
}

std::map<std::string, std::pair<ndt::type, const char *>> ndt::typevar_type::get_dynamic_type_properties() const {
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["name"] = {ndt::type("string"), reinterpret_cast<const char *>(&m_name)};

  return properties;
}

bool ndt::is_valid_typevar_name(const char *begin, const char *end) {
  if (begin != end) {
    if (*begin < 'A' || *begin > 'Z') {
      return false;
    }
    ++begin;
    while (begin < end) {
      char c = *begin;
      if ((c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') && c != '_') {
        return false;
      }
      ++begin;
    }
    return true;
  } else {
    return false;
  }
}
