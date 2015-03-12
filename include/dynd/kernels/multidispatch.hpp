//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <vector>

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/virtual.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct multidispatch_ck : virtual_ck {
      typedef std::unordered_map<std::vector<ndt::type>, arrfunc> map_type;

      struct data_type {
        std::shared_ptr<map_type> map;
        std::shared_ptr<std::vector<string>> vars;

        data_type(std::shared_ptr<map_type> map,
                  std::shared_ptr<std::vector<string>> vars)
            : map(map), vars(vars)
        {
        }
      };

      static intptr_t
      instantiate(const arrfunc_type_data *self, const arrfunc_type *af_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const array &kwds,
                  const std::map<string, ndt::type> &tp_vars);

      static void
      resolve_option_values(const arrfunc_type_data *self,
                            const arrfunc_type *af_tp, intptr_t nsrc,
                            const ndt::type *src_tp, nd::array &kwds,
                            const std::map<nd::string, ndt::type> &tp_vars);

      static int
      resolve_dst_type(const arrfunc_type_data *self, const arrfunc_type *af_tp,
                       intptr_t nsrc, const ndt::type *src_tp,
                       int throw_on_error, ndt::type &out_dst_tp,
                       const nd::array &kwds,
                       const std::map<nd::string, ndt::type> &tp_vars);
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd