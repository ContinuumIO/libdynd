//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    arrfunc neighborhood(const arrfunc &child);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd