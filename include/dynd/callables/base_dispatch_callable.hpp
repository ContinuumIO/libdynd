//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>

namespace dynd {
namespace nd {

  template <typename...>
  struct dispatch_callable;

  template <typename SpecializerType>
  struct dispatch_callable<SpecializerType> : base_callable {
    SpecializerType specializer;

    dispatch_callable(const ndt::type &tp, const SpecializerType &specializer)
        : base_callable(tp), specializer(specializer)
    {
    }

    const callable &specialize(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp)
    {
      return specializer(ret_tp, narg, arg_tp);
    }

    array alloc(const ndt::type *dst_tp) const { return empty(*dst_tp); }

    char *data_init(char *static_data, const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                    const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = const_cast<dispatch_callable *>(this)->specialize(dst_tp, nsrc, src_tp);

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();

      return child->data_init(static_data, child_dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp, intptr_t nsrc,
                          const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = const_cast<dispatch_callable *>(this)->specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        throw std::runtime_error("no suitable child for multidispatch");
      }

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();
      if (child_dst_tp.is_symbolic()) {
        child->resolve_dst_type(child->static_data(), data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      }
      else {
        dst_tp = child_dst_tp;
      }
    }

    void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = const_cast<dispatch_callable *>(this)->specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        std::stringstream ss;
        ss << "no suitable child for multidispatch for types " << src_tp[0] << ", and " << dst_tp << "\n";
        throw std::runtime_error(ss.str());
      }
      child->instantiate(child->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
                         kwds, tp_vars);
    }
  };

  template <typename OverloaderType, typename SpecializerType>
  struct dispatch_callable<OverloaderType, SpecializerType> : base_callable {
    SpecializerType specializer;
    OverloaderType overloader;

    dispatch_callable(const ndt::type &tp, const OverloaderType &overloader, const SpecializerType &specializer)
        : base_callable(tp), specializer(specializer), overloader(overloader)
    {
    }

    void overload(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp, const callable &value)
    {
      overloader(ret_tp, narg, arg_tp, value);
    }

    const callable &specialize(const ndt::type &ret_tp, intptr_t narg, const ndt::type *arg_tp)
    {
      return specializer(ret_tp, narg, arg_tp);
    }

    array alloc(const ndt::type *dst_tp) const { return empty(*dst_tp); }

    char *data_init(char *static_data, const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                    const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = const_cast<dispatch_callable *>(this)->specialize(dst_tp, nsrc, src_tp);

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();

      return child->data_init(static_data, child_dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp, intptr_t nsrc,
                          const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = const_cast<dispatch_callable *>(this)->specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        throw std::runtime_error("no suitable child for multidispatch");
      }

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();
      if (child_dst_tp.is_symbolic()) {
        child->resolve_dst_type(child->static_data(), data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      }
      else {
        dst_tp = child_dst_tp;
      }
    }

    void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = const_cast<dispatch_callable *>(this)->specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        std::stringstream ss;
        ss << "no suitable child for multidispatch for types " << src_tp[0] << ", and " << dst_tp << "\n";
        throw std::runtime_error(ss.str());
      }
      child->instantiate(child->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
                         kwds, tp_vars);
    }
  };

  struct base_dispatch_callable : base_callable {
    base_dispatch_callable(const ndt::type &tp) : base_callable(tp) {}

    array alloc(const ndt::type *dst_tp) const { return empty(*dst_tp); }

    char *data_init(char *static_data, const ndt::type &dst_tp, intptr_t nsrc, const ndt::type *src_tp, intptr_t nkwd,
                    const array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = specialize(dst_tp, nsrc, src_tp);

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();

      return child->data_init(static_data, child_dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
    }

    void resolve_dst_type(char *DYND_UNUSED(static_data), char *data, ndt::type &dst_tp, intptr_t nsrc,
                          const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        throw std::runtime_error("no suitable child for multidispatch");
      }

      const ndt::type &child_dst_tp = child.get_type()->get_return_type();
      if (child_dst_tp.is_symbolic()) {
        child->resolve_dst_type(child->static_data(), data, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      }
      else {
        dst_tp = child_dst_tp;
      }
    }

    void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      const callable &child = specialize(dst_tp, nsrc, src_tp);
      if (child.is_null()) {
        std::stringstream ss;
        ss << "no suitable child for multidispatch for types " << src_tp[0] << ", and " << dst_tp << "\n";
        throw std::runtime_error(ss.str());
      }
      child->instantiate(child->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd,
                         kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
