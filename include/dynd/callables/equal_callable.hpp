//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/comparison.hpp>
#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/equal_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class equal_callable : public default_instantiable_callable<equal_kernel<Arg0ID, Arg1ID>> {
  public:
    equal_callable()
        : default_instantiable_callable<equal_kernel<Arg0ID, Arg1ID>>(
              ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)})) {}
  };

  template <>
  class equal_callable<tuple_id, tuple_id> : public base_callable {
  public:
    struct equal_call_frame : call_frame {
      std::vector<uintptr_t> arrmeta_offsets;
    };

    equal_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<bool1>(), {ndt::type(tuple_id), ndt::type(tuple_id)}),
                        sizeof(equal_call_frame)) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      equal_call_frame *frame = reinterpret_cast<equal_call_frame *>(cg.back());
      frame->arrmeta_offsets = src_tp[0].extended<ndt::tuple_type>()->get_arrmeta_offsets();

      size_t field_count = src_tp[0].extended<ndt::tuple_type>()->get_field_count();
      for (size_t i = 0; i != field_count; ++i) {
        ndt::type child_src_tp[2] = {src_tp[0].extended<ndt::tuple_type>()->get_field_type(i),
                                     src_tp[1].extended<ndt::tuple_type>()->get_field_type(i)};
        if (!equal->is_abstract()) {
          cg.emplace_back(equal.get());
        }
        equal->new_resolve(this, cg, dst_tp, nsrc, child_src_tp, nkwd, kwds, tp_vars);
      }
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
      intptr_t self_offset = ckb.size();
      size_t field_count = reinterpret_cast<equal_call_frame *>(frame)->arrmeta_offsets.size();

      ckb.emplace_back<equal_kernel<tuple_id, tuple_id>>(kernreq, field_count,
                                                         reinterpret_cast<const uintptr_t *>(src_arrmeta[0]),
                                                         reinterpret_cast<const uintptr_t *>(src_arrmeta[1]));
      ckb.emplace_back(field_count * sizeof(size_t));

      equal_kernel<tuple_id, tuple_id> *self = ckb.get_at<equal_kernel<tuple_id, tuple_id>>(self_offset);
      const std::vector<uintptr_t> &arrmeta_offsets = reinterpret_cast<equal_call_frame *>(frame)->arrmeta_offsets;
      for (size_t i = 0; i != field_count; ++i) {
        self = ckb.get_at<equal_kernel<tuple_id, tuple_id>>(self_offset);
        size_t *field_kernel_offsets = self->get_offsets();
        field_kernel_offsets[i] = ckb.size() - self_offset;
        const char *child_src_arrmeta[2] = {src_arrmeta[0] + arrmeta_offsets[i], src_arrmeta[1] + arrmeta_offsets[i]};
        frame = frame->next();
        frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, dst_arrmeta, child_src_arrmeta,
                                       nkwd, kwds);
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
