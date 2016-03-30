//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/string_split_kernel.hpp>

namespace dynd {
namespace nd {

  class string_split_callable : public base_callable {
  public:
    string_split_callable()
        : base_callable(ndt::callable_type::make(ndt::make_var_dim(ndt::type(string_id)),
                                                 {ndt::type(string_id), ndt::type(string_id)})) {}

    void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                         const char *dst_arrmeta, const char *const *DYND_UNUSED(src_arrmeta), size_t DYND_UNUSED(nkwd),
                         const array *DYND_UNUSED(kwds)) {
      ckb.emplace_back<string_split_kernel>(
          kernreq, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref);
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(kwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<string_split_kernel>(
          kernreq, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref);
    }
  };

} // namespace dynd::nd
} // namespace dynd
