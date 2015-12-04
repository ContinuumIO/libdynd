//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/func/assignment.hpp>

namespace dynd {
namespace nd {

  struct DYND_API copy_kernel : base_virtual_kernel<copy_kernel> {
    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, intptr_t DYND_UNUSED(nkwd),
                                 const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0].get_canonical_type();
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      return assign::get()->instantiate(assign::get()->static_data(), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                        src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::copy_kernel> {
    static type make() { return type("(A... * S) -> B... * T"); }
  };

} // namespace dynd::ndt
} // namespace dynd
