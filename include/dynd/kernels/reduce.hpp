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
       * This is the resolution_data shared across the type resolution calling
       * sequence. It gets populated in resolve_option_values, and used by the
       * followup functions.
       */
      struct resolution_data_type {
        // Operator which does one step of reduction
        nd::arrfunc red_op;
        // NULL, one integer axis, list of integer axes, list of bool flag per axis
        nd::array axis;
        // To initialize a destination element from a source element
        nd::arrfunc dst_init;
        // Identity element of the reduction
        nd::array red_ident;
        // Properties red_op satisfies (TODO: should be metadata on red_op)
        bool associative;
        bool commutative;
        bool right_associative;
      };

      /**
       * Processes the input keyword arguments, and populates the
       * resolution_data struct in `resolution_data`.
       */
      static void
      resolve_option_values(const arrfunc_type_data *self,
                            const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                            char *resolution_data, intptr_t nsrc,
                            const ndt::type *src_tp, nd::array &kwds,
                            const std::map<nd::string, ndt::type> &tp_vars);

      /**
       * Instantiates the reduction into the ckernel_builder. The parameters to
       * the reduction are in the `resolution_data`.
       */
      size_t instantiate(const arrfunc_type_data *self,
                         const ndt::arrfunc_type *self_tp,
                         char *resolution_data, void *ckb, intptr_t ckb_offset,
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
