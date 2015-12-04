//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * Returns an arrfunc which copies data from one
   * array to another, without broadcasting
   */
  DYND_API extern struct copy : declfunc<copy> {
    DYND_API static callable make();
  } copy;

} // namespace dynd::nd
} // namespace dynd
