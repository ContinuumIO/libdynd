//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/base_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API escape_option_type : public base_type {
    type m_child_tp;

  public:
    escape_option_type(const type &child_tp);

    void print_type(std::ostream &o) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    bool operator==(const base_type &rhs) const;
  };

} // namespace dynd::ndt
} // namespace dynd
