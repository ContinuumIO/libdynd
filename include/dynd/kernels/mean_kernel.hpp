//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/arithmetic.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/types/any_kind_type.hpp>

namespace dynd {
namespace nd {

  // All methods are inlined, so this does not need to be declared DYND_API.
  struct mean_kernel : base_strided_kernel<mean_kernel, 1> {
    std::intptr_t compound_div_offset;
    int64 count;

    mean_kernel(int64 count) : count(count) {}

    void single(char *dst, char *const *src)
    {
      kernel_prefix *sum_kernel = get_child();
      kernel_single_t sum = sum_kernel->get_function<kernel_single_t>();
      sum(sum_kernel, dst, src);

      kernel_prefix *compound_div_kernel = get_child(compound_div_offset);
      kernel_single_t compound_div = compound_div_kernel->get_function<kernel_single_t>();
      char *child_src[1] = {reinterpret_cast<char *>(&count)};
      compound_div(compound_div_kernel, dst, child_src);
    }
  };

} // namespace dynd::nd
} // namespace dynd
