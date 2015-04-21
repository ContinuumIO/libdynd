//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct reduce_virtual_ck : base_virtual_kernel<reduce_virtual_ck> {

      /**
       * Instantiates a reduction. The parameters to the reduction are
       * provided as keyword parameters
       *
       * op: arrfunc     The arrfunc being reduced. Must be either binary or
       *                 unary.
       * axis: N * bool  N is the number of dimensions to reduce, and this
       *                 array is true for the particular axes.
       * dst_init: arrfunc  If not NULL, provides an arrfunc to initialize an
       *                    accumulator output from an input value.
       * red_ident: Any     If not NULL, provides an identity element to copy.
       * associative: bool   If true, `op` is associative.
       * commutative: bool   If true, `op` is commutative.
       * right_associative: bool   If true, `op` is right associative, otherwise
       *                           it is left associative. This is for a unary
       *                           `op`, and when `associative` is false.
       */
      size_t instantiate(const arrfunc_type_data *self,
                         const ndt::arrfunc_type *self_tp, char *data,
                         void *ckb, intptr_t ckb_offset,
                         const ndt::type &dst_tp, const char *dst_arrmeta,
                         intptr_t nsrc, const ndt::type *src_tp,
                         const char *const *src_arrmeta,
                         kernel_request_t kernreq,
                         const eval::eval_context *ectx, const nd::array &kwds,
                         const std::map<dynd::nd::string, ndt::type> &tp_vars);
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
