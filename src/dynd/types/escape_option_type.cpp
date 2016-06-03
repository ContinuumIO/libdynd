//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/escape_option_type.hpp>

using namespace std;
using namespace dynd;

ndt::escape_option_type::escape_option_type(const type &child_tp)
    : base_type(escape_option_id, 0, 1, child_tp.get_flags() | type_flag_symbolic, 0, 0, 0), m_child_tp(child_tp) {}

void ndt::escape_option_type::print_type(ostream &o) const { o << "\\?" << m_child_tp; }

bool ndt::escape_option_type::match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const {
  const type &child_element_tp = m_child_tp.extended<base_dim_type>()->get_element_type();
  return m_child_tp.match(candidate_tp, tp_vars) || child_element_tp.match(candidate_tp, tp_vars);
}

bool ndt::escape_option_type::operator==(const base_type &rhs) const {
  return (this == &rhs || rhs.get_id() == escape_option_id) &&
         m_child_tp == reinterpret_cast<const escape_option_type &>(rhs).m_child_tp;
}
