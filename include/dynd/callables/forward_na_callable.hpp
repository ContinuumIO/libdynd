//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/forward_na_kernel.hpp>
#include <dynd/option.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <size_t... I>
    class forward_na_callable : public base_callable {
      callable m_child;

    public:
      forward_na_callable(const ndt::type &child_tp) : base_callable(ndt::make_type<forward_na_callable>(child_tp)) {}

      forward_na_callable(const callable &child)
          : base_callable(ndt::make_type<forward_na_callable>(child.get_type())), m_child(child) {}

      ndt::type resolve(base_callable *caller, char *DYND_UNUSED(data), call_graph &cg, const ndt::type &dst_tp,
                        size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        base_callable *child = m_child ? m_child.get() : caller;

        cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc,
                           const char *const *src_arrmeta) {
          size_t self_offset = kb.size();
          kb.emplace_back<forward_na_kernel<index_sequence<I...>>>(kernreq);

          kb(kernel_request_single, dst_arrmeta, nsrc, src_arrmeta);

          for (size_t i : {I...}) {
            size_t is_na_offset = kb.size() - self_offset;
            kb(kernel_request_single, nullptr, 1, src_arrmeta + i);
            kb.get_at<forward_na_kernel<index_sequence<I...>>>(self_offset)->is_na_offset[i] = is_na_offset;
          }

          size_t assign_na_offset = kb.size() - self_offset;
          kb(kernel_request_single, nullptr, 0, nullptr);
          kb.get_at<forward_na_kernel<index_sequence<I...>>>(self_offset)->assign_na_offset = assign_na_offset;
        });

        ndt::type src_value_tp[2];
        for (size_t i = 0; i < 2; ++i) {
          src_value_tp[i] = src_tp[i];
        }
        for (size_t i : {I...}) {
          src_value_tp[i] = src_value_tp[i].extended<ndt::option_type>()->get_value_type();
        }

        ndt::type res_value_tp =
            child->resolve(this, nullptr, cg, dst_tp.is_symbolic() ? child->get_return_type() : dst_tp, 2, src_value_tp,
                           nkwd, kwds, tp_vars);

        for (size_t i : {I...}) {
          is_na->resolve(this, nullptr, cg, ndt::make_type<bool>(), 1, src_tp + i, 0, nullptr, tp_vars);
        }

        return assign_na->resolve(this, nullptr, cg, ndt::make_type<ndt::option_type>(res_value_tp), 0, nullptr, nkwd,
                                  kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd

namespace ndt {

  template <size_t... I>
  struct traits<nd::functional::forward_na_callable<I...>> {
    static type equivalent(const type &child_tp) {
      const type &ret_tp = child_tp.extended<callable_type>()->get_return_type();
      const std::vector<type> &arg_tp = child_tp.extended<callable_type>()->get_pos_types();

      return callable_type::make(make_type<option_type>(ret_tp), arg_tp);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
