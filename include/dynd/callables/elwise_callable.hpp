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
      elwise_callable() : base_callable(ndt::type(), sizeof(elwise_call_frame)) {}

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

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic() ||
            child_tp->get_return_type().get_id() == typevar_constructed_id) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size, dst_stride;
        std::array<intptr_t, N> src_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          intptr_t src_size;
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            finished &= src_ndim == 1;
          } else {
            std::stringstream ss;
            ss << "make_elwise_strided_dimension_expr_kernel: expected strided "
                  "or fixed dim, got "
               << src_tp[i];
            throw std::runtime_error(ss.str());
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, fixed_dim_id, N>>(kernreq, size, dst_stride, src_stride.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }

        // Instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      virtual void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                               const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                               intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}
    };

    template <size_t N>
    class elwise_callable<fixed_dim_id, var_dim_id, N> {
    public:
      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        intptr_t size, dst_stride;
        if (!dst_tp.get_as_strided(dst_arrmeta, &size, &dst_stride, &child_dst_tp, &child_dst_arrmeta)) {
          std::stringstream ss;
          ss << "make_elwise_strided_dimension_expr_kernel: error processing "
                "type "
             << dst_tp << " as strided";
          throw type_error(ss.str());
        }

        std::array<intptr_t, N> src_stride, src_offset;
        std::array<bool, N> is_src_var;
        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          intptr_t src_size;
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          // The src[i] strided parameters
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size, &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            // Check for a broadcasting error
            if (src_size != 1 && size != src_size) {
              throw broadcast_error(dst_tp, dst_arrmeta, src_tp[i], src_arrmeta[i]);
            }
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        ckb->emplace_back<elwise_kernel<fixed_dim_id, var_dim_id, N>>(kernreq, size, dst_stride, src_stride.data(),
                                                                      src_offset.data(), is_src_var.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // Instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
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

      elwise_callable() : base_callable(ndt::type(), sizeof(elwise_call_frame)) {}

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

        /*
                std::array<intptr_t, N> src_arrmeta_offsets;
                for (size_t i = 0; i < N; ++i) {
                  src_arrmeta_offsets[i] = stack.arg_metadata_offsets()[i];
                  if (data.is_src_var[i]) {
                    src_arrmeta_offsets[i] += sizeof(ndt::var_dim_type::metadata_type);
                  } else {
                    src_arrmeta_offsets[i] += sizeof(size_stride_t);
                  }
                }
        */

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

      virtual void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                                   const char *dst_arrmeta, const char *const *src_arrmeta, size_t nkwd,
                                   const array *kwds) {
        intptr_t target_alignment = reinterpret_cast<elwise_call_frame *>(frame)->target_alignment;

        const ndt::var_dim_type::metadata_type *dst_md =
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

        std::array<intptr_t, N> src_stride;
        std::array<intptr_t, N> src_offset;
        std::array<intptr_t, N> src_size;
        for (size_t i = 0; i < N; ++i) {
          if (reinterpret_cast<elwise_call_frame *>(frame)->broadcast_src[i] &&
              !reinterpret_cast<elwise_call_frame *>(frame)->is_src_var[i]) {
            src_stride[i] = 0;
            src_offset[i] = 0;
            src_size[i] = 1;
          } else if (reinterpret_cast<elwise_call_frame *>(frame)->is_src_var[i]) {
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);

            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
          } else {
            src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
            src_offset[i] = 0;
          }
        }

        std::array<const char *, N> child_src_arrmeta;
        for (size_t i = 0; i < N; ++i) {
          child_src_arrmeta[i] = src_arrmeta[i];
          if (reinterpret_cast<elwise_call_frame *>(frame)->is_src_var[i]) {
            child_src_arrmeta[i] += sizeof(ndt::var_dim_type::metadata_type);
          } else {
            child_src_arrmeta[i] += sizeof(size_stride_t);
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

      static void elwise_instantiate(callable &self, callable &child, char *data, kernel_builder *ckb,
                                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                     intptr_t nkwd, const nd::array *kwds,
                                     const std::map<std::string, ndt::type> &tp_vars) {
        const ndt::callable_type *child_tp = child.get_type();

        intptr_t dst_ndim = dst_tp.get_ndim();
        if (!child_tp->get_return_type().is_symbolic()) {
          dst_ndim -= child_tp->get_return_type().get_ndim();
        }

        const char *child_dst_arrmeta;
        std::array<const char *, N> child_src_arrmeta;
        ndt::type child_dst_tp;
        std::array<ndt::type, N> child_src_tp;

        // The dst var parameters
        const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();
        const ndt::var_dim_type::metadata_type *dst_md =
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);

        child_dst_arrmeta = dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type);
        child_dst_tp = dst_vdd->get_element_type();

        std::array<intptr_t, N> src_stride, src_offset, src_size;
        std::array<bool, N> is_src_var;

        bool finished = dst_ndim == 1;
        for (size_t i = 0; i < N; ++i) {
          // The src[i] strided parameters
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          if (src_ndim < dst_ndim) {
            // This src value is getting broadcasted
            src_stride[i] = 0;
            src_offset[i] = 0;
            src_size[i] = 1;
            is_src_var[i] = false;
            child_src_arrmeta[i] = src_arrmeta[i];
            child_src_tp[i] = src_tp[i];
            finished &= src_ndim == 0;
          } else if (src_tp[i].get_as_strided(src_arrmeta[i], &src_size[i], &src_stride[i], &child_src_tp[i],
                                              &child_src_arrmeta[i])) {
            src_offset[i] = 0;
            is_src_var[i] = false;
            finished &= src_ndim == 1;
          } else {
            const ndt::var_dim_type *vdd = static_cast<const ndt::var_dim_type *>(src_tp[i].extended());
            const ndt::var_dim_type::metadata_type *src_md =
                reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);
            src_stride[i] = src_md->stride;
            src_offset[i] = src_md->offset;
            is_src_var[i] = true;
            child_src_arrmeta[i] = src_arrmeta[i] + sizeof(ndt::var_dim_type::metadata_type);
            child_src_tp[i] = vdd->get_element_type();
            finished &= src_ndim == 1;
          }
        }

        ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
            kernreq, dst_md->blockref.get(), dst_vdd->get_target_alignment(), dst_md->stride, dst_md->offset,
            src_stride.data(), src_offset.data(), src_size.data(), is_src_var.data());

        // If there are still dimensions to broadcast, recursively lift more
        if (!finished) {
          return self->instantiate(data, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                   child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
        }
        // All the types matched, so instantiate the elementwise handler
        return child->instantiate(NULL, ckb, child_dst_tp, child_dst_arrmeta, nsrc, child_src_tp.data(),
                                  child_src_arrmeta.data(), kernel_request_strided, nkwd, kwds, tp_vars);
      }

      virtual void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                               const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                               intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                               const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                               intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                               const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {}

      /*
            virtual void new_instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char
         *dst_arrmeta,
                                         intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                         const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t
         DYND_UNUSED(nkwd),
                                         const array *DYND_UNUSED(kwds))
            {
              const ndt::var_dim_type::metadata_type *dst_md =
                  reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta);
              const ndt::var_dim_type *dst_vdd = dst_tp.extended<ndt::var_dim_type>();

              std::array<intptr_t, N> src_stride;
              std::array<intptr_t, N> src_offset;
              std::array<intptr_t, N> src_size;
              for (size_t i = 0; i < N; ++i) {
                if (reinterpret_cast<data_type *>(data)->broadcast_src[i] &&
                    !reinterpret_cast<data_type *>(data)->is_src_var[i]) {
                  src_stride[i] = 0;
                  src_offset[i] = 0;
                  src_size[i] = 1;
                }
                else if (reinterpret_cast<data_type *>(data)->is_src_var[i]) {
                  const ndt::var_dim_type::metadata_type *src_md =
                      reinterpret_cast<const ndt::var_dim_type::metadata_type *>(src_arrmeta[i]);

                  src_stride[i] = src_md->stride;
                  src_offset[i] = src_md->offset;
                }
                else {
                  src_stride[i] = reinterpret_cast<const size_stride_t *>(src_arrmeta[i])->stride;
                  src_offset[i] = 0;
                }
              }

              ckb->emplace_back<elwise_kernel<var_dim_id, fixed_dim_id, N>>(
                  kernreq, dst_md->blockref.get(), dst_vdd->get_target_alignment(), dst_md->stride, dst_md->offset,
                  src_stride.data(), src_offset.data(), src_size.data(),
                  reinterpret_cast<data_type *>(data)->is_src_var.data());
            }
      */
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
