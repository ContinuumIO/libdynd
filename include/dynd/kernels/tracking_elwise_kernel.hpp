//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/elwise_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <size_t N>
    struct tracking_kernel : base_strided_kernel<tracking_kernel<N>, N> {
      //    size_t ndim;

      //      tracking_kernel(size_t ndim = 2) : ndim(ndim) {}

      ~tracking_kernel() { this->get_child()->destroy(); }

      void single(char *dst, char *const *src) {
        std::cout << "tracking_kernel<" << N << ">::single" << std::endl;

        char *child_src[N + 1];
        for (size_t i = 0; i < N; ++i) {
          child_src[i] = src[i];
        }

        iteration_t it;
        it.ndim = 2;
        it.index = new size_t[10];
        it.index[0] = 0;
        child_src[N] = reinterpret_cast<char *>(&it);

        this->get_child()->single(dst, child_src);
      }
    };

    struct X_prefix {
      size_t index_offset;
    };

    template <type_id_t RetID, type_id_t ArgID, size_t N>
    struct tracking_elwise_kernel;

    template <size_t N>
    struct tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>
        : base_elwise_kernel<tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N>, X_prefix {
      size_t index_offset;

      tracking_elwise_kernel(size_t size, intptr_t dst_stride, const intptr_t *src_stride)
          : base_elwise_kernel<tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N>(size, dst_stride, src_stride) {
      }

      void single(char *dst, char *const *src) {
        std::cout << "tracking_elwise_kernel::single" << std::endl;

        reinterpret_cast<iteration_t *>(src[2])->index[1] = 0;
        base_elwise_kernel<tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N>::single(dst, src);
      }

      size_t &begin(char *const *src) {
        std::cout << "tracking_elwise_kernel::begin" << std::endl;
        return reinterpret_cast<iteration_t *>(src[2])->index[0];
      }
    };

    template <type_id_t RetID, type_id_t ArgID, size_t N>
    struct inner_tracking_elwise_kernel;

    template <size_t N>
    struct inner_tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>
        : base_elwise_kernel<inner_tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N + 1> {
      size_t ndim;

      inner_tracking_elwise_kernel(size_t size, intptr_t dst_stride, const intptr_t *src_stride, size_t ndim)
          : base_elwise_kernel<inner_tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N + 1>(size, dst_stride,
                                                                                                   src_stride),
            ndim(ndim) {
        this->m_src_stride[N] = 0;
        std::cout << "inner_tracking_elwise_kernel::constructor" << std::endl;
        std::cout << "ndim = " << ndim << std::endl;
      }

      kernel_prefix *get_child() {
        return kernel_prefix::get_child(
            kernel_builder::aligned_size(sizeof(inner_tracking_elwise_kernel) + ndim * sizeof(size_t)));
      }

      void call(array *dst, const array *src) {
        char *src_data[N];
        for (size_t i = 0; i < N; ++i) {
          src_data[i] = const_cast<char *>(src[i].cdata());
        }
        single(const_cast<char *>(dst->cdata()), src_data);
      }

      void single(char *dst, char *const *src) {
        iteration_t it{ndim,
                       reinterpret_cast<size_t *>(reinterpret_cast<char *>(this) +
                                                  kernel_builder::aligned_size(sizeof(inner_tracking_elwise_kernel)))};
        it.index[0] = 0;
        std::cout << "inner_tracking_elwise_kernel::single" << std::endl;

        char *child_src[N + 1];
        for (size_t i = 0; i < N; ++i) {
          child_src[i] = src[i];
        }
        child_src[N] = reinterpret_cast<char *>(&it);

        it.index[1] = 0;
        base_elwise_kernel<inner_tracking_elwise_kernel<fixed_dim_id, fixed_dim_id, N>, N + 1>::single(dst, child_src);
      }

      size_t &begin(char *const *src) {
        std::cout << "inner_tracking_elwise_kernel::begin" << std::endl;
        return reinterpret_cast<iteration_t *>(src[N + 1])->index[0];
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
