//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Lifts the provided arrfunc, broadcasting it as necessary to execute
     * across additional dimensions in the arguments.
     *
     * \param child  The arrfunc being lifted
     */
    arrfunc elwise(const arrfunc &child);

    arrfunc elwise(const ndt::type &self_tp, const arrfunc &child);

    ndt::type elwise_make_type(const ndt::arrfunc_type *child_tp);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
