//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/byteswap_kernels.hpp>

namespace dynd {
namespace nd {

  class byteswap_callable : public base_callable {
  public:
    struct byteswap_call_frame : call_frame {
      size_t data_size;
    };

    byteswap_callable() : base_callable(ndt::type("(Any) -> Any"), sizeof(byteswap_call_frame)) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &DYND_UNUSED(dst_tp),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t DYND_UNUSED(nkwd),
                     const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      byteswap_call_frame *data = reinterpret_cast<byteswap_call_frame *>(cg.back());
      data->data_size = src_tp[0].get_data_size();
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds)) {
      ckb.emplace_back<byteswap_ck>(kernreq, reinterpret_cast<byteswap_call_frame *>(frame)->data_size);
    }
  };

  class pairwise_byteswap_callable : public base_callable {
  public:
    struct pairwise_byteswap_call_frame : call_frame {
      size_t data_size;
    };

    pairwise_byteswap_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &DYND_UNUSED(dst_tp),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t DYND_UNUSED(nkwd),
                     const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      pairwise_byteswap_call_frame *data = reinterpret_cast<pairwise_byteswap_call_frame *>(cg.back());
      data->data_size = src_tp[0].get_data_size();
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds)) {
      ckb.emplace_back<pairwise_byteswap_ck>(kernreq,
                                             reinterpret_cast<pairwise_byteswap_call_frame *>(frame)->data_size);
    }
  };

} // namespace dynd::nd
} // namespace dynd
