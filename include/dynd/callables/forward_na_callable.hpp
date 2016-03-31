//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/forward_na_kernel.hpp>

namespace dynd {
namespace nd {

  template <int... I>
  class forward_na_callable;

  template <int I>
  class forward_na_callable<I> : public base_callable {
    callable m_child;

  public:
    forward_na_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      std::cout << "forward_na_callable::new_resolve" << std::endl;

      ndt::type child_src_tp[2];
      child_src_tp[I] = src_tp[I].extended<ndt::option_type>()->get_value_type();
      child_src_tp[1 - I] = src_tp[1 - I];

      if (!is_na->is_abstract()) {
        cg.emplace_back(is_na.get());
      }
      is_na->new_resolve(this, cg, dst_tp, nsrc, src_tp + I, nkwd, kwds, tp_vars);

      if (!assign_na->is_abstract()) {
        cg.emplace_back(assign_na.get());
      }
      ndt::type assign_na_dst_tp = ndt::make_type<ndt::option_type>(ndt::make_type<bool1>());
      assign_na->new_resolve(this, cg, assign_na_dst_tp, 0, nullptr, nkwd, kwds, tp_vars);

      const ndt::type &child_dst_tp = m_child.get_ret_type();
      if (!m_child->is_abstract()) {
        cg.emplace_back(m_child.get());
      }
      if (dst_tp.is_symbolic()) {
        dst_tp = child_dst_tp;
        m_child->new_resolve(this, cg, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      } else {
        ndt::type dst_tp_to_resolve = dst_tp;
        m_child->new_resolve(this, cg, dst_tp_to_resolve, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      }
      dst_tp = ndt::make_type<ndt::option_type>(dst_tp);
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
      size_t self_offset = ckb.size();
      size_t child_offsets[2];

      ckb.emplace_back<forward_na_kernel<I>>(kernreq);
      ckb.emplace_back(2 * sizeof(size_t));

      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, dst_arrmeta, src_arrmeta + I, nkwd, kwds);

      child_offsets[0] = ckb.size() - self_offset;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, nullptr, nullptr, nkwd, kwds);

      child_offsets[1] = ckb.size() - self_offset;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, dst_arrmeta, src_arrmeta, nkwd, kwds);

      memcpy(ckb.get_at<forward_na_kernel<I>>(self_offset)->get_offsets(), child_offsets, 2 * sizeof(size_t));
    }
  };

  template <callable &Callable, bool Src0IsOption, bool Src1IsOption>
  class option_comparison_callable;

  template <callable &Callable>
  class option_comparison_callable<Callable, true, true> : public base_callable {
  public:
    option_comparison_callable() : base_callable(ndt::type("(?Scalar, ?Scalar) -> ?bool")) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      std::cout << "option_comparison_callable::new_resolve" << std::endl;

      if (!is_na->is_abstract()) {
        cg.emplace_back(is_na.get());
      }
      is_na->new_resolve(this, cg, dst_tp, nsrc, &src_tp[0], nkwd, kwds, tp_vars);

      if (!is_na->is_abstract()) {
        cg.emplace_back(is_na.get());
      }
      is_na->new_resolve(this, cg, dst_tp, nsrc, &src_tp[1], nkwd, kwds, tp_vars);

      const ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::option_type>()->get_value_type(),
                                         src_tp[1].extended<ndt::option_type>()->get_value_type()};
      if (!Callable->is_abstract()) {
        cg.emplace_back(Callable.get());
      }
      ndt::type child_dst_tp = dst_tp.extended<ndt::option_type>()->get_value_type();
      Callable->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);

      if (!assign_na->is_abstract()) {
        cg.emplace_back(assign_na.get());
      }
      ndt::type assign_na_dst_tp = ndt::make_type<ndt::option_type>(ndt::make_type<bool1>());
      assign_na->new_resolve(this, cg, assign_na_dst_tp, 0, nullptr, nkwd, kwds, tp_vars);
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
      intptr_t ckb_offset = ckb.size();
      intptr_t option_comp_offset = ckb_offset;
      ckb.emplace_back<option_comparison_kernel<true, true>>(kernreq);
      ckb_offset = ckb.size();

      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, dst_arrmeta, &src_arrmeta[0], nkwd, kwds);
      ckb_offset = ckb.size();
      option_comparison_kernel<true, true> *self = ckb.get_at<option_comparison_kernel<true, true>>(option_comp_offset);
      self->is_na_rhs_offset = ckb_offset - option_comp_offset;

      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, dst_arrmeta, &src_arrmeta[1], nkwd, kwds);
      ckb_offset = ckb.size();
      self = ckb.get_at<option_comparison_kernel<true, true>>(option_comp_offset);
      self->comp_offset = ckb_offset - option_comp_offset;
      auto cmp = Callable;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, dst_arrmeta, src_arrmeta, nkwd, kwds);
      ckb_offset = ckb.size();
      self = ckb.get_at<option_comparison_kernel<true, true>>(option_comp_offset);
      self->assign_na_offset = ckb_offset - option_comp_offset;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernel_request_single, nullptr, nullptr, nkwd, kwds);
      ckb_offset = ckb.size();
    }
  };

} // namespace dynd::nd
} // namespace dynd
