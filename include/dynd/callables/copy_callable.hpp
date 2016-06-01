//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/assignment.hpp>

namespace dynd {
namespace nd {

  class copy_callable : public base_callable {
  public:
    copy_callable() : base_callable(ndt::type("(A... * S) -> B... * T")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &tp_vars) {
      array error_mode = eval::default_eval_context.errmode;
      assign->resolve(this, nullptr, cg, dst_tp, 1, src_tp, 1, &error_mode, tp_vars);
      return src_tp[0].get_canonical_type();
    }
  };

} // namespace dynd::nd
} // namespace dynd
