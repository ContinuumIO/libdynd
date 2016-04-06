//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename I>
    struct forward_na_kernel;

    template <size_t... I>
    struct forward_na_kernel<index_sequence<I...>> : base_strided_kernel<forward_na_kernel<index_sequence<I...>>, 2> {
      size_t is_na_offset[2];
      size_t assign_na_offset;

      void single(char *res, char *const *args) {
        for (intptr_t i : {I...}) {
          bool1 is_na;
          this->get_child(is_na_offset[i])->single(reinterpret_cast<char *>(&is_na), args + i);

          // Check if args[I] is not available
          if (is_na) {
            // assign_na
            return this->get_child(assign_na_offset)->single(res, nullptr);
          }
        }

        // call the actual child
        this->get_child()->single(res, args);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
