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
  extern struct copy : declfunc<copy> {
    static callable make();
  } copy;

  /**
   * Returns an arrfunc which copies data from one
   * array to another, with broadcasting.
   */
  extern struct broadcast_copy : declfunc<broadcast_copy> {
    static callable make();
  } broadcast_copy;

} // namespace dynd::nd
} // namespace dynd
