//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

bool ndt::scalar_kind_type::operator==(const base_type &other) const
{
  return this == &other || other.get_id() == scalar_kind_id;
}

bool ndt::scalar_kind_type::match(const type &candidate_tp, std::map<std::string, type> &DYND_UNUSED(tp_vars)) const
{
  // Match against any scalar
  return candidate_tp.is_scalar();
}

void ndt::scalar_kind_type::print_type(ostream &o) const { o << "Scalar"; }
