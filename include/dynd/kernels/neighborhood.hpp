//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/substitute_shape.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <int N>
    struct neighborhood_kernel : base_kernel<neighborhood_kernel<N>, N> {
      typedef neighborhood_kernel<N> self_type;

      struct data_type {
        intptr_t ndim;
        intptr_t *shape;
        int *offset;

        data_type(intptr_t ndim, intptr_t *shape, int *offset) : ndim(ndim), shape(shape), offset(offset)
        {
        }
      };

      intptr_t dst_stride;
      intptr_t src0_offset;
      intptr_t src0_stride;
      intptr_t count[3];
      intptr_t nh_size;
      intptr_t src0_size;
      intptr_t nh_offset;

      neighborhood_kernel(intptr_t dst_size, intptr_t dst_stride, intptr_t src0_size, intptr_t src0_stride,
                          intptr_t size, intptr_t offset)
          : dst_stride(dst_stride), src0_offset(offset * src0_stride), src0_stride(src0_stride), nh_size(size),
            src0_size(src0_size), nh_offset(offset)
      {
        count[0] = -offset;
        if (count[0] < 0) {
          count[0] = 0;
        } else if (count[0] > dst_size) {
          count[0] = dst_size;
        }

        count[2] = size + offset - 1;
        if (count[2] < 0) {
          count[2] = 0;
        } else if (count[2] > (dst_size - count[0])) {
          count[2] = dst_size - count[0];
        }

        count[1] = dst_size - count[0] - count[2];
      }

      void single(char *, char *const *)
      {
        //        ckernel_prefix *child = self_type::get_child();
        //      char *src0_copy = src[0] + src0_offset;

        // 0 < i < src0_size
        // offset < i + offset < src0_size + offset
        // offset < i + offset

        std::cout << "left" << std::endl;
        intptr_t i = nh_offset;
        while (i < std::min((intptr_t)0, src0_size + nh_offset)) {
          std::cout << i << std::endl;
          ++i;
        };

        std::cout << "middle" << std::endl;
        while (i < std::min(src0_size + nh_offset, src0_size - nh_size + 1)) {
          std::cout << i << std::endl;
          ++i;
        }

        std::cout << "right" << std::endl;
        while (i < (src0_size + nh_offset)) {
          std::cout << i << std::endl;
          ++i;
        }

        /*
                nh_start_stop->start = count[0];
                nh_start_stop->stop = nh_size; // min(nh_size, dst_size)
                for (intptr_t i = 0; i < count[0]; ++i) {
                  child_fn(child, dst, src_copy);
                  --(nh_start_stop->start);
                  dst += dst_stride;
                  for (intptr_t j = 0; j < N; ++j) {
                    src_copy[j] += src_stride[j];
                  }
                }
        */

        //  *nh_start = 0;
        //    *nh_stop = nh_size;
        //        for (intptr_t i = 0; i < count[1]; ++i) {
        //        child->single(dst, &src0_copy);
        //      dst += dst_stride;
        //    src0_copy += src0_stride;
        //  }

        /*
                for (intptr_t i = 0; i < count[2]; ++i) {
                  --(nh_start_stop->stop);
                  child_fn(child, dst, src_copy);
                  dst += dst_stride;
                  for (intptr_t j = 0; j < N; ++j) {
                    src_copy[j] += src_stride[j];
                  }
                }
        */
      }
      static void data_init(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data,
                            const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                            const ndt::type *DYND_UNUSED(src_tp), intptr_t DYND_UNUSED(nkwd), const array *kwds,
                            const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        new (data) data_type(kwds[0].get_dim_size(), reinterpret_cast<intptr_t *>(kwds[0].get_readwrite_originptr()),
                             kwds[1].is_missing() ? NULL : reinterpret_cast<int *>(kwds[1].get_readwrite_originptr()));
      }

      static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
                                   char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                                   const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // swap in the input dimension values for the Fixed**N
        intptr_t ndim = src_tp[0].get_ndim();
        dimvector shape(ndim);
        src_tp[0].extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
        dst_tp = ndt::substitute_shape(dst_tp, ndim, shape.get());
      }

      static intptr_t instantiate(char *static_data, size_t data_size, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        neighborhood_kernel::make(
            ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(dst_arrmeta)->dim_size,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(dst_arrmeta)->stride,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride,
            reinterpret_cast<data_type *>(data)->shape[0],
            (reinterpret_cast<data_type *>(data)->offset == NULL) ? 0 : reinterpret_cast<data_type *>(data)->offset[0]);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        const char *child_dst_arrmeta = dst_arrmeta + sizeof(fixed_dim_type_arrmeta);

        ndt::type child_src_tp[N];
        const char *child_src_arrmeta[N];
        for (int i = 0; i < N; ++i) {
          child_src_tp[i] = src_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
          child_src_arrmeta[i] = src_arrmeta[i] + sizeof(fixed_dim_type_arrmeta);
        }

        reinterpret_cast<data_type *>(data)->ndim -= 1;
        reinterpret_cast<data_type *>(data)->shape += 1;
        if (reinterpret_cast<data_type *>(data)->offset != NULL) {
          reinterpret_cast<data_type *>(data)->offset += 1;
        }

        if (reinterpret_cast<data_type *>(data)->ndim == 0) {
          reinterpret_cast<data_type *>(data)->~data_type();

          const callable &child = *reinterpret_cast<callable *>(static_data);
          return child.get()->instantiate(child.get()->static_data, child.get()->data_size, NULL, ckb, ckb_offset,
                                          child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp, child_src_arrmeta,
                                          kernel_request_single, ectx, nkwd - 3, kwds + 3, tp_vars);
        }

        return instantiate(static_data, data_size, data, ckb, ckb_offset, child_dst_tp, child_dst_arrmeta, nsrc,
                           child_src_tp, child_src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
