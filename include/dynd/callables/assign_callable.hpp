//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd {
namespace nd {

  template <type_id_t ResID, type_id_t Arg0ID>
  class assign_callable : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(ResID), {ndt::type(Arg0ID)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        ckb.emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                   assign_error_nocheck>>(kernreq);
        break;
      case assign_error_overflow:
        ckb.emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                   assign_error_overflow>>(kernreq);
        break;
      case assign_error_fractional:
        ckb.emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                   assign_error_fractional>>(kernreq);
        break;
      case assign_error_inexact:
        ckb.emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                   assign_error_inexact>>(kernreq);
        break;
      default:
        throw std::runtime_error("error in assign_callable::instantiate");
      }
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                    assign_error_nocheck>>(kernreq);
        break;
      case assign_error_overflow:
        ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                    assign_error_overflow>>(kernreq);
        break;
      case assign_error_fractional:
        ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                    assign_error_fractional>>(kernreq);
        break;
      case assign_error_inexact:
        ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                                    assign_error_inexact>>(kernreq);
        break;
      default:
        throw std::runtime_error("error in assign_callable::instantiate");
      }
    }
  };

  template <>
  class assign_callable<bool_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(bool_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      case assign_error_overflow:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_overflow>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      case assign_error_fractional:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_fractional>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      case assign_error_inexact:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_inexact>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      default:
        throw std::runtime_error("error");
      }
    }
  };

  template <>
  class assign_callable<fixed_bytes_id, fixed_bytes_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_bytes_id), {ndt::type(fixed_bytes_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb), const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t DYND_UNUSED(kernreq), intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      throw std::runtime_error("cannot assign to a fixed_bytes type of a different size");
    }
  };

  template <type_id_t IntID>
  class int_to_string_assign_callable : public base_callable {
  public:
    int_to_string_assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(IntID), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *src_arrmeta, size_t DYND_UNUSED(nkwd),
                         const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      ndt::type src_tp(string_id);
      ckb.emplace_back<detail::assignment_kernel<IntID, int_kind_id, string_id, string_kind_id, assign_error_default>>(
          kernreq, src_tp, src_arrmeta[0], error_mode);
    }
  };

  template <type_id_t IntID>
  class string_to_int_assign_callable : public base_callable {
  public:
    string_to_int_assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(IntID)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *src_arrmeta, size_t DYND_UNUSED(nkwd),
                         const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      ndt::type src_tp(string_id);
      ckb.emplace_back<detail::assignment_kernel<IntID, int_kind_id, string_id, string_kind_id, assign_error_default>>(
          kernreq, src_tp, src_arrmeta[0], error_mode);
    }
  };

  template <>
  class assign_callable<float64_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(float64_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      ckb->emplace_back<
          detail::assignment_kernel<float64_id, float_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, src_tp[0], src_arrmeta[0], error_mode);
    }
  };

  template <>
  class assign_callable<fixed_string_id, string_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      size_t dst_data_size;
      string_encoding_t dst_encoding;
      string_encoding_t src0_encoding;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_string_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();

      assign_call_frame *data = reinterpret_cast<assign_call_frame *>(cg.back());
      data->dst_encoding = dst_tp.extended<ndt::fixed_string_type>()->get_encoding();
      data->dst_data_size = dst_tp.get_data_size();
      data->src0_encoding = src_fs->get_encoding();
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      ckb.emplace_back<
          detail::assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq,
          get_next_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->src0_encoding, error_mode),
          get_append_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->dst_encoding, error_mode),
          reinterpret_cast<assign_call_frame *>(frame)->dst_data_size, error_mode != assign_error_nocheck);
    }
  };

  template <>
  class assign_callable<fixed_string_id, fixed_string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_string_id), {ndt::type(fixed_string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      const ndt::fixed_string_type *src_fs = src_tp[0].extended<ndt::fixed_string_type>();
      ckb->emplace_back<detail::assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id,
                                                  assign_error_nocheck>>(
          kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), error_mode),
          get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(), error_mode),
          dst_tp.get_data_size(), src_fs->get_data_size(), error_mode != assign_error_nocheck);
    }
  };

  template <>
  class assign_callable<string_id, int_kind_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(int_kind_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<
          detail::assignment_kernel<string_id, string_kind_id, int8_id, int_kind_id, assign_error_nocheck>>(
          kernreq, dst_tp, src_tp[0].get_id(), dst_arrmeta);
    }
  };

  template <>
  class assign_callable<string_id, char_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      size_t src0_data_size;
      string_encoding_t dst_encoding;
      string_encoding_t src0_encoding;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(char_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_call_frame *data = reinterpret_cast<assign_call_frame *>(cg.back());
      data->dst_encoding = dst_tp.extended<ndt::base_string_type>()->get_encoding();
      data->src0_data_size = src_tp[0].get_data_size();
      data->src0_encoding = src_tp[0].extended<ndt::char_type>()->get_encoding();
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      ckb.emplace_back<
          detail::assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, reinterpret_cast<assign_call_frame *>(frame)->dst_encoding,
          reinterpret_cast<assign_call_frame *>(frame)->src0_encoding,
          reinterpret_cast<assign_call_frame *>(frame)->src0_data_size,
          get_next_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->src0_encoding, error_mode),
          get_append_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->dst_encoding,
                                                error_mode));
    }
  };

  template <>
  class assign_callable<type_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(type_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *src_arrmeta, size_t DYND_UNUSED(nkwd),
                         const array *DYND_UNUSED(kwds)) {
      ckb.emplace_back<
          detail::assignment_kernel<type_id, scalar_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, ndt::type(string_id), src_arrmeta[0]);
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<
          detail::assignment_kernel<type_id, scalar_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, src_tp[0], src_arrmeta[0]);
    }
  };

  template <>
  class assign_callable<string_id, fixed_string_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      size_t src0_data_size;
      string_encoding_t dst_encoding;
      string_encoding_t src0_encoding;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(fixed_string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();

      assign_call_frame *data = reinterpret_cast<assign_call_frame *>(cg.back());
      data->dst_encoding = dst_tp.extended<ndt::base_string_type>()->get_encoding();
      data->src0_data_size = src_tp[0].get_data_size();
      data->src0_encoding = src_fs->get_encoding();
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      ckb.emplace_back<
          detail::assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, reinterpret_cast<assign_call_frame *>(frame)->dst_encoding,
          reinterpret_cast<assign_call_frame *>(frame)->src0_encoding,
          reinterpret_cast<assign_call_frame *>(frame)->src0_data_size,
          get_next_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->src0_encoding, error_mode),
          get_append_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->dst_encoding,
                                                error_mode));
    }
  };

  template <>
  class assign_callable<float32_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(float32_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      ckb->emplace_back<
          detail::assignment_kernel<float32_id, float_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, src_tp[0], src_arrmeta[0], error_mode);
    }
  };

  template <>
  class assign_callable<string_id, type_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(type_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                         const char *dst_arrmeta, const char *const *DYND_UNUSED(src_arrmeta), size_t DYND_UNUSED(nkwd),
                         const array *DYND_UNUSED(kwds)) {
      ckb.emplace_back<
          detail::assignment_kernel<string_id, string_kind_id, type_id, scalar_kind_id, assign_error_nocheck>>(
          kernreq, ndt::type(string_id), dst_arrmeta);
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<
          detail::assignment_kernel<string_id, string_kind_id, type_id, scalar_kind_id, assign_error_nocheck>>(
          kernreq, dst_tp, dst_arrmeta);
    }
  };

  template <>
  class assign_callable<char_id, string_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      size_t dst_data_size;
      string_encoding_t dst_encoding;
      string_encoding_t src0_encoding;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(char_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();

      assign_call_frame *data = reinterpret_cast<assign_call_frame *>(cg.back());
      data->dst_encoding = dst_tp.extended<ndt::char_type>()->get_encoding();
      data->dst_data_size = dst_tp.get_data_size();
      data->src0_encoding = src_fs->get_encoding();
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq,
                         const char *DYND_UNUSED(dst_arrmeta), const char *const *DYND_UNUSED(src_arrmeta),
                         size_t DYND_UNUSED(nkwd), const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      ckb.emplace_back<
          detail::assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq,
          get_next_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->src0_encoding, error_mode),
          get_append_unicode_codepoint_function(reinterpret_cast<assign_call_frame *>(frame)->dst_encoding, error_mode),
          reinterpret_cast<assign_call_frame *>(frame)->dst_data_size, error_mode != assign_error_nocheck);
    }
  };

  template <>
  class assign_callable<pointer_id, pointer_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(pointer_id), {ndt::type(pointer_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      ckb->emplace_back<assignment_kernel<pointer_id, pointer_id>>(kernreq);

      const char *child_src_arrmeta = src_arrmeta[0] + sizeof(pointer_type_arrmeta);
      assign->instantiate(NULL, ckb, dst_tp.extended<ndt::pointer_type>()->get_target_type(), dst_arrmeta, 1,
                          &src_tp[0].extended<ndt::pointer_type>()->get_target_type(), &child_src_arrmeta,
                          kernel_request_single, nkwd, kwds, tp_vars);
    }
  };

  template <>
  class assign_callable<option_id, option_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(option_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      if (!is_na->is_abstract()) {
        cg.emplace_back(is_na.get());
      }
      ndt::type is_na_dst_tp = ndt::make_type<bool1>();
      is_na->new_resolve(this, cg, is_na_dst_tp, 1, src_tp, nkwd, kwds, tp_vars);

      if (!assign_na->is_abstract()) {
        cg.emplace_back(assign_na.get());
      }
      assign_na->new_resolve(this, cg, dst_tp, 0, nullptr, nkwd, kwds, tp_vars);

      ndt::type dst_val_tp = dst_tp.extended<ndt::option_type>()->get_value_type();
      const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
      if (!assign->is_abstract()) {
        cg.emplace_back(assign.get());
      }
      assign->new_resolve(this, cg, dst_val_tp, 1, &src_val_tp, nkwd, kwds, tp_vars);
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
      intptr_t ckb_offset = ckb.size();
      intptr_t root_ckb_offset = ckb_offset;
      typedef detail::assignment_kernel<option_id, any_kind_id, option_id, any_kind_id, assign_error_nocheck> self_type;
      ckb.emplace_back<self_type>(kernreq);
      ckb_offset = ckb.size();
      // instantiate src_is_avail
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, NULL, src_arrmeta, nkwd, kwds);
      ckb_offset = ckb.size();
      // instantiate dst_assign_na
      ckb.reserve(ckb_offset + sizeof(kernel_prefix));
      self_type *self = ckb.get_at<self_type>(root_ckb_offset);
      self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, dst_arrmeta, NULL, nkwd, kwds);
      ckb_offset = ckb.size();
      // instantiate value_assign
      ckb.reserve(ckb_offset + sizeof(kernel_prefix));
      self = ckb.get_at<self_type>(root_ckb_offset);
      self->m_value_assign_offset = ckb_offset - root_ckb_offset;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, dst_arrmeta, src_arrmeta, nkwd,
                                     kwds);
    }
  };

  template <>
  class assign_callable<option_id, float_kind_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(float_kind_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {
      m_abstract = true;
    }

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      static callable f = make_callable<assign_callable<option_id, option_id>>();

      // Deal with some float32 to option[T] conversions where any NaN is
      // interpreted
      // as NA.
      ndt::type src_tp_as_option = ndt::make_type<ndt::option_type>(src_tp[0]);
      if (!f->is_abstract()) {
        cg.emplace_back(f.get());
      }
      f->new_resolve(this, cg, dst_tp, 1, &src_tp_as_option, nkwd, kwds, tp_vars);
    }
  };

  template <>
  class assign_callable<option_id, string_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      type_id_t tid;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}),
              sizeof(assign_call_frame)) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      assign_call_frame *data = reinterpret_cast<assign_call_frame *>(cg.back());
      data->tid = dst_tp.extended<ndt::option_type>()->get_value_type().get_id();

      if (data->tid == string_id) {
        // Just a string to string assignment
        if (!assign->is_abstract()) {
          cg.emplace_back(assign.get());
        }
        ndt::type new_dst_tp = dst_tp.extended<ndt::option_type>()->get_value_type();
        assign->new_resolve(this, cg, new_dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

        return;
      }
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      // Deal with some string to option[T] conversions where string values
      // might mean NA
      type_id_t tid = reinterpret_cast<assign_call_frame *>(frame)->tid;
      switch (tid) {
      case bool_id:
        ckb.emplace_back<detail::string_to_option_bool_ck>(kernreq);
        return;
      case int8_id:
      case int16_id:
      case int32_id:
      case int64_id:
      case int128_id:
      case float16_id:
      case float32_id:
      case float64_id:
        ckb.emplace_back<detail::string_to_option_number_ck>(kernreq, tid, error_mode);
        return;
      case string_id: {
        frame = frame->next();
        frame->callee->new_instantiate(frame, ckb, kernreq, dst_arrmeta, src_arrmeta, nkwd, kwds);
        return;
      }
      default:
        break;
      }

      // Fall back to an adaptor that checks for a few standard
      // missing value tokens, then uses the standard value assignment
      intptr_t ckb_offset = ckb.size();
      intptr_t root_ckb_offset = ckb_offset;
      ckb.emplace_back<detail::string_to_option_tp_ck>(kernreq);
      ckb_offset = ckb.size();
      // First child ckernel is the value assignment
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, dst_arrmeta, src_arrmeta, nkwd,
                                     kwds);

      ckb_offset = ckb.size();
      // Re-acquire self because the address may have changed
      detail::string_to_option_tp_ck *self = ckb.get_at<detail::string_to_option_tp_ck>(root_ckb_offset);
      // Second child ckernel is the NA assignment
      self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, dst_arrmeta, src_arrmeta, nkwd,
                                     kwds);

      ckb_offset = ckb.size();
    }
  };

  template <>
  class assign_callable<tuple_id, tuple_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      ndt::type dst_tp;
      ndt::type src0_tp;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(tuple_id), {ndt::type(tuple_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}),
              sizeof(assign_call_frame)) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_call_frame *frame = reinterpret_cast<assign_call_frame *>(cg.back());
      frame->dst_tp = dst_tp;
      frame->src0_tp = src_tp[0];
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds)) {
      ndt::type dst_tp = reinterpret_cast<assign_call_frame *>(frame)->dst_tp;
      ndt::type src0_tp = reinterpret_cast<assign_call_frame *>(frame)->src0_tp;
      reinterpret_cast<assign_call_frame *>(frame)->~assign_call_frame();

      if (dst_tp.extended() == src0_tp.extended()) {
        make_tuple_identical_assignment_kernel(&ckb, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq);
        return;
      } else if (src0_tp.get_id() == tuple_id || src0_tp.get_id() == struct_id) {
        make_tuple_assignment_kernel(&ckb, dst_tp, dst_arrmeta, src0_tp, src_arrmeta[0], kernreq);
        return;
      } else if (src0_tp.is_builtin()) {
        make_broadcast_to_tuple_assignment_kernel(&ckb, dst_tp, dst_arrmeta, src0_tp, src_arrmeta[0], kernreq);
        return;
      }

      std::stringstream ss;
      ss << "Cannot assign from " << src0_tp << " to " << dst_tp;
      throw dynd::type_error(ss.str());
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      if (dst_tp.extended() == src_tp[0].extended()) {
        make_tuple_identical_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq);
      } else if (src_tp[0].get_id() == tuple_id || src_tp[0].get_id() == struct_id) {
        make_tuple_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq);
      } else if (src_tp[0].is_builtin()) {
        make_broadcast_to_tuple_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq);
      } else {
        std::stringstream ss;
        ss << "Cannot assign from " << src_tp[0] << " to " << dst_tp;
        throw dynd::type_error(ss.str());
      }
    }
  };

  template <>
  class assign_callable<struct_id, struct_id> : public base_callable {
  public:
    struct assign_call_frame : call_frame {
      ndt::type dst_tp;
      ndt::type src0_tp;
    };

    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(struct_id), {ndt::type(struct_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}),
              sizeof(assign_call_frame)) {}

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_call_frame *frame = reinterpret_cast<assign_call_frame *>(cg.back());
      frame->dst_tp = dst_tp;
      frame->src0_tp = src_tp[0];
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds)) {
      ndt::type dst_tp = reinterpret_cast<assign_call_frame *>(frame)->dst_tp;
      ndt::type src0_tp = reinterpret_cast<assign_call_frame *>(frame)->src0_tp;
      reinterpret_cast<assign_call_frame *>(frame)->~assign_call_frame();

      if (dst_tp.extended() == src0_tp.extended()) {
        make_tuple_identical_assignment_kernel(&ckb, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq);
        return;
      } else if (src0_tp.get_id() == struct_id) {
        make_struct_assignment_kernel(&ckb, dst_tp, dst_arrmeta, src0_tp, src_arrmeta[0], kernreq);
        return;
      } else if (src0_tp.is_builtin()) {
        make_broadcast_to_tuple_assignment_kernel(&ckb, dst_tp, dst_arrmeta, src0_tp, src_arrmeta[0], kernreq);
        return;
      }

      std::stringstream ss;
      ss << "Cannot assign from " << src0_tp << " to " << dst_tp;
      throw dynd::type_error(ss.str());
    }
  };

  class option_to_value_callable : public base_callable {
  public:
    option_to_value_callable() : base_callable(ndt::type("(Any) -> Any")) { m_new_style = true; }

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      if (!is_na->is_abstract()) {
        cg.emplace_back(is_na.get());
      }
      ndt::type is_na_dst_tp = ndt::make_type<bool1>();
      is_na->new_resolve(this, cg, is_na_dst_tp, 1, src_tp, nkwd, kwds, tp_vars);

      const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
      if (!assign->is_abstract()) {
        cg.emplace_back(assign.get());
      }
      assign->new_resolve(this, cg, dst_tp, 1, &src_val_tp, nkwd, kwds, tp_vars);
    }

    void new_instantiate(call_frame *frame, kernel_builder &ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                         const char *const *src_arrmeta, size_t nkwd, const array *kwds) {
      intptr_t ckb_offset = ckb.size();
      intptr_t root_ckb_offset = ckb_offset;
      typedef dynd::nd::option_to_value_ck self_type;
      ckb.emplace_back<self_type>(kernreq);
      // instantiate src_is_na
      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, NULL, src_arrmeta, nkwd, kwds);
      ckb_offset = ckb.size();
      // instantiate value_assign
      ckb.reserve(ckb_offset + sizeof(kernel_prefix));
      self_type *self = ckb.get_at<self_type>(root_ckb_offset);
      self->m_value_assign_offset = ckb_offset - root_ckb_offset;

      frame = frame->next();
      frame->callee->new_instantiate(frame, ckb, kernreq | kernel_request_data_only, dst_arrmeta, src_arrmeta, nkwd,
                                     kwds);
    }
  };

  class adapt_assign_from_callable : public base_callable {
  public:
    adapt_assign_from_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      intptr_t ckb_offset = ckb->size();
      const ndt::type &storage_tp = src_tp[0].storage_type();
      if (storage_tp.is_expression()) {
        const callable &forward = src_tp[0].extended<ndt::adapt_type>()->get_forward();

        intptr_t self_offset = ckb_offset;
        ckb->emplace_back<detail::adapt_assign_from_kernel>(kernreq, storage_tp.get_canonical_type());
        ckb_offset = ckb->size();

        nd::assign->instantiate(data, ckb, storage_tp.get_canonical_type(), dst_arrmeta, nsrc, &storage_tp, src_arrmeta,
                                kernel_request_single, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        intptr_t forward_offset = ckb_offset - self_offset;
        ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
        forward->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp2, src_arrmeta, kernel_request_single, nkwd,
                             kwds, tp_vars);
        ckb_offset = ckb->size();
        ckb->get_at<detail::adapt_assign_from_kernel>(self_offset)->forward_offset = forward_offset;
      } else {
        const callable &forward = src_tp[0].extended<ndt::adapt_type>()->get_forward();

        ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
        forward->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp2, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    }
  };

  class adapt_assign_to_callable : public base_callable {
  public:
    adapt_assign_to_callable() : base_callable(ndt::type("(Any) -> Any")) { m_abstract = true; }

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t nsrc,
                     const ndt::type *DYND_UNUSED(src_tp), size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      const callable &inverse = dst_tp.extended<ndt::adapt_type>()->get_inverse();
      if (!inverse->is_abstract()) {
        cg.emplace_back(inverse.get());
      }

      ndt::type storage_tp = dst_tp.storage_type();
      const ndt::type &value_tp = dst_tp.value_type();
      inverse->new_resolve(this, cg, storage_tp, nsrc, &value_tp, nkwd, kwds, tp_vars);
    }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const ndt::type *DYND_UNUSED(src_tp), const char *const *src_arrmeta, kernel_request_t kernreq,
                     intptr_t nkwd, const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      const callable &inverse = dst_tp.extended<ndt::adapt_type>()->get_inverse();
      const ndt::type &value_tp = dst_tp.value_type();
      inverse->instantiate(data, ckb, dst_tp.storage_type(), dst_arrmeta, nsrc, &value_tp, src_arrmeta, kernreq, nkwd,
                           kwds, tp_vars);
    }
  };

  class assignment_option_callable : public base_callable {
  public:
    assignment_option_callable() : base_callable(ndt::type("(Any) -> ?Any")) { m_abstract = true; }

    void new_resolve(base_callable *DYND_UNUSED(parent), call_graph &cg, ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, size_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      ndt::type val_dst_tp =
          dst_tp.get_id() == option_id ? dst_tp.extended<ndt::option_type>()->get_value_type() : dst_tp;
      ndt::type val_src_tp =
          src_tp[0].get_id() == option_id ? src_tp[0].extended<ndt::option_type>()->get_value_type() : src_tp[0];

      if (!assign->is_abstract()) {
        cg.emplace_back(assign.get());
      }
      assign->new_resolve(this, cg, val_dst_tp, 1, &val_src_tp, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
