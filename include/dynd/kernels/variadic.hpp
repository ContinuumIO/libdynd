//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/virtual.hpp>

namespace dynd {
namespace nd {

  template <template <int> class CKT>
  struct variadic_ck : virtual_ck<variadic_ck<CKT>> {
    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq, const eval::eval_context *ectx,
                const dynd::nd::array &kwds,
                const std::map<dynd::nd::string, ndt::type> &tp_vars)
    {
      switch (nsrc) {
      case 0:
        return CKT<0>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      case 1:
        return CKT<1>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      case 2:
        return CKT<2>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      case 3:
        return CKT<3>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      case 4:
        return CKT<4>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      case 5:
        return CKT<5>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      case 6:
        return CKT<6>::instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                                   dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq, ectx, kwds, tp_vars);
      default:
        throw std::runtime_error("ckernel with nsrc > 6 not implemented");
      }
    }

    static void
    resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                     char *data, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, const dynd::nd::array &kwds,
                     const std::map<dynd::nd::string, ndt::type> &tp_vars)

    {
      switch (nsrc) {
      case 0:
        return CKT<0>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      case 1:
        return CKT<1>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      case 2:
        return CKT<2>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      case 3:
        return CKT<3>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      case 4:
        return CKT<4>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      case 5:
        return CKT<5>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      case 6:
        return CKT<6>::resolve_dst_type(self, self_tp, data, dst_tp, nsrc,
                                        src_tp, kwds, tp_vars);
      default:
        throw std::runtime_error("ckernel with nsrc > 6 not implemented");
      }
    }

    static void resolve_option_values(
        const arrfunc_type_data *self, const arrfunc_type *self_tp, char *data,
        intptr_t nsrc, const ndt::type *src_tp, nd::array &kwds,
        const std::map<nd::string, ndt::type> &tp_vars)
    {
      switch (nsrc) {
      case 0:
        return CKT<0>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      case 1:
        return CKT<1>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      case 2:
        return CKT<2>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      case 3:
        return CKT<3>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      case 4:
        return CKT<4>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      case 5:
        return CKT<5>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      case 6:
        return CKT<6>::resolve_option_values(self, self_tp, data, nsrc, src_tp,
                                             kwds, tp_vars);
      default:
        throw std::runtime_error("error");
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd