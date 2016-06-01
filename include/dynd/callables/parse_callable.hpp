//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/parse_kernel.hpp>

namespace dynd {
namespace nd {
  namespace json {

    template <typename ReturnType>
    class parse_callable : public default_instantiable_callable<parse_kernel<ReturnType>> {
    public:
      parse_callable()
          : default_instantiable_callable<parse_kernel<ReturnType>>(ndt::make_type<ndt::callable_type>(
                ndt::make_type<ReturnType>(), {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}
    };

    template <>
    class parse_callable<ndt::option_type> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::make_type<ndt::callable_type>(
                ndt::make_type<ndt::option_type>(ndt::make_type<ndt::any_kind_type>()),
                {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
          intptr_t self_offset = kb.size();
          kb.emplace_back<parse_kernel<ndt::option_type>>(kernreq);

          kb(kernreq | kernel_request_data_only, nullptr, dst_arrmeta, 0, nullptr);
          kb.get_at<parse_kernel<ndt::option_type>>(self_offset)->parse_offset = kb.size() - self_offset;
          kb(kernreq | kernel_request_data_only, nullptr, dst_arrmeta, nsrc, src_arrmeta);
        });

        assign_na->resolve(this, nullptr, cg, dst_tp, 0, nullptr, nkwd, kwds, tp_vars);
        dynamic_parse->resolve(this, nullptr, cg, dst_tp.extended<ndt::option_type>()->get_value_type(), nsrc, src_tp,
                               nkwd, kwds, tp_vars);

        return dst_tp;
      }

      /*
            void instantiate(call_node *&DYND_UNUSED(node), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                             const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
         *src_arrmeta,
                             kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                             const std::map<std::string, ndt::type> &tp_vars) {
              intptr_t ckb_offset = ckb->size();
              intptr_t self_offset = ckb_offset;
              ckb->emplace_back<parse_kernel<option_id>>(kernreq);
              ckb_offset = ckb->size();

              assign_na->instantiate(node, data, ckb, dst_tp, dst_arrmeta, 0, nullptr, nullptr,
                                     kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
              ckb_offset = ckb->size();

              ckb->get_at<parse_kernel<option_id>>(self_offset)->parse_offset = ckb_offset - self_offset;
              dynamic_parse->instantiate(node, data, ckb, dst_tp.extended<ndt::option_type>()->get_value_type(),
         dst_arrmeta,
                                         nsrc, src_tp, src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds,
         tp_vars);
              ckb_offset = ckb->size();
            }
      */
    };

    template <>
    class parse_callable<ndt::struct_type> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::struct_type>(),
                                                             {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        return dst_tp;
      }

      /*
            void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                             const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
         *src_arrmeta,
                             kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                             const std::map<std::string, ndt::type> &tp_vars) {
              intptr_t ckb_offset = ckb->size();
              size_t field_count = dst_tp.extended<ndt::struct_type>()->get_field_count();
              const std::vector<uintptr_t> &arrmeta_offsets =
         dst_tp.extended<ndt::struct_type>()->get_arrmeta_offsets();

              intptr_t self_offset = ckb_offset;
              ckb->emplace_back<parse_kernel<struct_id>>(kernreq, dst_tp, field_count,
                                                         dst_tp.extended<ndt::struct_type>()->get_data_offsets(dst_arrmeta));
              node = next(node);

              ckb_offset = ckb->size();

              for (size_t i = 0; i < field_count; ++i) {
                ckb->get_at<parse_kernel<struct_id>>(self_offset)->child_offsets[i] = ckb_offset - self_offset;
                dynamic_parse->instantiate(node, data, ckb, dst_tp.extended<ndt::struct_type>()->get_field_type(i),
                                           dst_arrmeta + arrmeta_offsets[i], nsrc, src_tp, src_arrmeta, kernreq, nkwd,
         kwds,
                                           tp_vars);
                ckb_offset = ckb->size();
              }
            }
      */
    };

    template <>
    class parse_callable<ndt::fixed_dim_type> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::make_type<ndt::callable_type>(
                ndt::make_type<ndt::fixed_dim_kind_type>(ndt::make_type<ndt::any_kind_type>()),
                {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        return dst_tp;
      }

      /*
            void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                             const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
         *src_arrmeta,
                             kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                             const std::map<std::string, ndt::type> &tp_vars) {
              ckb->emplace_back<parse_kernel<fixed_dim_id>>(kernreq, dst_tp,
                                                            reinterpret_cast<const size_stride_t
         *>(dst_arrmeta)->dim_size,
                                                            reinterpret_cast<const size_stride_t
         *>(dst_arrmeta)->stride);
              node = next(node);

              const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
              dynamic_parse->instantiate(node, data, ckb, child_dst_tp,
                                         dst_arrmeta + sizeof(ndt::fixed_dim_type::metadata_type), nsrc, src_tp,
         src_arrmeta,
                                         kernreq, nkwd, kwds, tp_vars);
            }
      */
    };

    template <>
    class parse_callable<ndt::var_dim_type> : public base_callable {
    public:
      parse_callable()
          : base_callable(ndt::make_type<ndt::callable_type>(
                ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::any_kind_type>()),
                {ndt::make_type<char *>(), ndt::make_type<char *>()})) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        return dst_tp;
      }

      /*
            void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                             const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
         *src_arrmeta,
                             kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                             const std::map<std::string, ndt::type> &tp_vars) {
              ckb->emplace_back<parse_kernel<var_dim_id>>(
                  kernreq, dst_tp, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref,
                  reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride);
              node = next(node);

              const ndt::type &child_dst_tp = dst_tp.extended<ndt::var_dim_type>()->get_element_type();
              dynamic_parse->instantiate(node, data, ckb, child_dst_tp,
                                         dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), nsrc, src_tp,
         src_arrmeta,
                                         kernreq, nkwd, kwds, tp_vars);
            }
      */
    };

  } // namespace dynd::nd::json
} // namespace dynd::nd
} // namespace dynd
