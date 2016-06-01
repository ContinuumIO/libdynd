//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/cbrt_callable.hpp>
#include <dynd/unary_arithmetic.hpp>

template <typename>
using isdef_cbrt = std::true_type;

DYND_API nd::callable nd::cbrt = make_unary_arithmetic<nd::cbrt_callable, isdef_cbrt, type_sequence<float, double>>();
