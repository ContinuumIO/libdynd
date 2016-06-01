//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/serialize_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  class serialize_callable : public base_callable {
  public:
    serialize_callable()
        : base_callable(ndt::make_type<ndt::callable_type>(ndt::type("bytes"), {ndt::make_type<Arg0Type>()})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      size_t data_size = src_tp[0].get_data_size();
      cg.emplace_back([data_size](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                  const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                  const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<serialize_kernel<Arg0Type>>(kernreq, data_size);
      });

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
