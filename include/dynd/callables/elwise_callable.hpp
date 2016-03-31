//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <array>

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/elwise_kernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <type_id_t DstTypeID, type_id_t SrcTypeID, size_t N>
    class elwise_callable;

    template <size_t N>
    class elwise_callable<fixed_dim_id, fixed_dim_id, N> : public base_callable {
      struct elwise_call_frame : call_frame {
        bool broadcast_dst;
        std::array<bool, N> broadcast_src;
      };

    public:
      elwise_callable() : base_callable(ndt::type(), sizeof(elwise_call_frame)) { m_new_style = true; }

      callable &get_child(base_callable *parent);

      void new_resolve(base_callable *parent, call_graph &cg, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                       size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(cg.back());

        callable &child = get_child(parent);
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_id() == typevar_constructed_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        ndt::type child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        std::array<ndt::type, N> child_src_tp;

        intptr_t size;
        size = dst_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          intptr_t src_size = src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            data->broadcast_src[i] = true;
            //            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else {
            data->broadcast_src[i] = false;
            src_size = src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
            child_src_tp[i] = src_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
            if (src_size != 1 && size != src_size) {
              throw std::runtime_error("broadcast error");
            }

            finished &= src_ndim == 1;
          }
        }

        if (!finished) {
          if (!parent->is_abstract()) {
            cg.emplace_back(parent);
          }

          parent->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        } else {
          if (!child->is_abstract()) {
            cg.emplace_back(child.get());
          }

          child->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        }
      }

      void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                           const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(frame);

        intptr_t size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
        intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride;
        for (size_t i = 0; i < N; ++i) {
          if (data->broadcast_src[i]) {
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
          } else {
            src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(size_stride_t);
          }
        }

        ckb.emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        frame = frame->next();
        frame->callee->new_instantiate(frame, ckb, kernel_request_strided, dst_arrmeta + sizeof(size_stride_t),
                                       child_src_arrmeta.data(), nkwd, kwds);
      }
    };

    template <size_t N>
    class elwise_callable<fixed_dim_id, var_dim_id, N> : public base_callable {
    public:
      struct elwise_call_frame : call_frame {
        std::array<bool, N> broadcast_src;
        std::array<bool, N> is_src_var;
      };

    public:
      elwise_callable() : base_callable(ndt::type(), sizeof(elwise_call_frame)) {}

      callable &get_child(base_callable *parent);

      void new_resolve(base_callable *parent, call_graph &cg, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                       size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(cg.back());

        callable &child = get_child(parent);

        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size;
        size = dst_tp.extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
        child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_size;
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          // The src[i] strided parameters
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            data->broadcast_src[i] = true;
            data->is_src_var[i] = false;
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_id() == fixed_dim_id) {
            // Check for a broadcasting error
            src_size = src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size();
            child_src_tp[i] = src_tp[i].extended<ndt::fixed_dim_type>()->get_element_type();
            if (src_size != 1 && size != src_size) {
              throw std::runtime_error("broadcast error");
              //              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            //            src_offset[i] = 0;
            data->is_src_var[i] = false;
            data->broadcast_src[i] = false;
            finished &= src_ndim == 1;
          } else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            data->is_src_var[i] = true;
            data->broadcast_src[i] = false;
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          if (!parent->is_abstract()) {
            cg.emplace_back(parent);
          }

          return parent->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        } else {
          if (!child->is_abstract()) {
            cg.emplace_back(parent);
          }

          return child->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        }
      }

      void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                           const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(frame);

        intptr_t dst_size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
        intptr_t dst_stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride, src_offset;
        for (size_t i = 0; i < N; ++i) {
          if (data->broadcast_src[i] && !data->is_src_var[i]) {
            src_stride[i] = 0;
            src_offset[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
          } else if (!data->broadcast_src[i] && !data->is_src_var[i]) {
            src_offset[i] = 0;
          } else {
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_offset[i] = src_md->offset;
            src_stride[i] = src_md->stride;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
          }
        }
        ckb.emplace_back<elwise_kernel<fixed_dim_id, var_dim_id, N>>(kernreq, dst_size, dst_stride, src_stride.data(),
                                                                     src_offset.data(), data->is_src_var.data());

        frame = frame->next();
        frame->callee->new_instantiate(frame, ckb, kernel_request_strided, dst_arrmeta + sizeof(size_stride_t),
                                       child_src_arrmeta.data(), nkwd, kwds);
      }
    };

    template <size_t N>
    class elwise_callable<var_dim_id, fixed_dim_id, N> : public base_callable {
    public:
      struct elwise_call_frame : call_frame {
        intptr_t target_alignment;
        bool broadcast_dst;
        std::array<bool, N> broadcast_src;
        std::array<bool, N> is_src_var;
      };

      elwise_callable() : base_callable(ndt::type(), sizeof(elwise_call_frame)) { m_new_style = true; }

      callable &get_child(base_callable *parent);

      void new_resolve(base_callable *parent, call_graph &cg, ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp,
                       size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {

        elwise_call_frame *data = reinterpret_cast<elwise_call_frame *>(cg.back());
        data->target_alignment = dst_tp.extended<ndt::var_dim_type>()->get_target_alignment();

        callable &child = get_child(parent);

        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        // The dst var parameters
        const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();

        child_dst_tp = dst_vdd->get_element_type();

        std::array<intptr_t, N> src_size;

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          // The src[i] strided parameters
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_size[i] = 1;
            data->broadcast_src[i] = true;
            data->is_src_var[i] = false;
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_id() == fixed_dim_id) { // src_tp[i].get_as_strided(src_arrmeta[i], &src_size[i],
                                                           // &src_stride[i], &child_src_tp[i],
            //                      &child_src_arrmeta[i])) {
            child_src_tp[i] = src_tp[i].extended<ndt::base_dim_type>()->get_element_type();
            data->broadcast_src[i] = false;
            data->is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            data->is_src_var[i] = true;
            data->broadcast_src[i] = true;
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          if (!parent->is_abstract()) {
            cg.emplace_back(parent);
          }

          /*
                    stack.push_back(parent, child_dst_tp, stack.res_metadata_offset() +
             sizeof(ndt::var_dim_type::metadata_type),
                                    stack.narg(), child_src_tp.data(), src_arrmeta_offsets.data(),
             kernel_request_strided);
          */

          parent->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);
        } else {
          if (!child->is_abstract()) {
            cg.emplace_back(child.get());
          }
          // All the types matched, so instantiate the elementwise handler
          //          stack.push_back(m_child, child_dst_tp, stack.res_metadata_offset() +
          //          sizeof(ndt::var_dim_type::metadata_type),
          //                        stack.narg(), child_src_tp.data(), src_arrmeta_offsets.data(),
          //                        kernel_request_strided);

          child->new_resolve(this, cg, child_dst_tp, nsrc, child_src_tp.data(), nkwd, kwds, tp_vars);

          //          return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc,
          //         child_src_tp.data(),
          //                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds,
          //       tp_vars);
        }
      }

      void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                           const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
        intptr_t target_alignment = reinterpret_cast<elwise_call_frame *>(frame)->target_alignment;

        const ndt::var_dim_type::metadata_type *dst_md =
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

        std::array<const char *, N> child_src_arrmeta;
        std::array<intptr_t, N> src_stride;
        std::array<intptr_t, N> src_offset;
        std::array<intptr_t, N> src_size;
        for (size_t i = 0; i < N; ++i) {
          if (reinterpret_cast<elwise_call_frame *>(frame)->broadcast_src[i] &&
              !reinterpret_cast<elwise_call_frame *>(frame)->is_src_var[i]) {
            src_stride[i] = 0;
            src_offset[i] = 0;
            src_size[i] = 1;
            child_src_arrmeta[i] = src_arrmeta[i];
          } else if (!reinterpret_cast<elwise_call_frame *>(frame)->broadcast_src[i] &&
                     !reinterpret_cast<elwise_call_frame *>(frame)->is_src_var[i]) {
            src_offset[i] = 0;
            src_size[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size;
            src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->stride;

          } else {
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);

            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
          }
        }

        ckb.emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
            kernreq, dst_md->blockref.get(), target_alignment, dst_md->stride, dst_md->offset, src_stride.data(),
            src_offset.data(), src_size.data(), reinterpret_cast<elwise_call_frame *>(frame)->is_src_var.data());

        frame = frame->next();
        frame->callee->new_instantiate(frame, ckb, kernel_request_strided,
                                       dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), child_src_arrmeta.data(),
                                       nkwd, kwds);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
