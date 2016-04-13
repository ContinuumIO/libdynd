//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/callables/elwise_callable.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/iteration_type.hpp>
#include <dynd/kernels/tracking_elwise_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <size_t N>
    class elwise_iteration_callable : public base_callable {
      callable m_child;

    public:
      elwise_iteration_callable(const ndt::type &tp, const callable &child) : base_callable(tp), m_child(child) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                        const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
        std::cout << "elwise_iteration_callable<" << N << ">::resolve" << std::endl;

        cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, const char *dst_arrmeta,
                           size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          std::cout << "elwise_iteration_callable<" << N << ">::instantiate" << std::endl;

          kb.emplace_back<elwise_iteration_kernel<N>>(kernreq);

          kb(kernreq | kernel_request_data_only, dst_arrmeta, N + 1, src_arrmeta);
        });

        ndt::type child_src_tp[N + 1];
        for (size_t i = 0; i < N; ++i) {
          child_src_tp[i] = src_tp[i];
        }
        child_src_tp[N] = ndt::make_type<ndt::iteration_type>();

        return m_child->resolve(this, nullptr, cg, dst_tp, N + 1, child_src_tp, nkwd, kwds, tp_vars);
      }
    };

    /**
     * This defines the type and keyword argument resolution for
     * an elwise callable.
     */
    template <size_t N>
    class elwise_dispatch_callable : public base_callable {
    public:
      struct data_type {
        base_callable *child;
        bool tracking;
        size_t ndim;
      };

      callable m_child;
      bool m_tracking;

      elwise_dispatch_callable(const ndt::type &tp, const callable &child, bool tracking = false)
          : base_callable(tp), m_child(child), m_tracking(tracking) {}

      ndt::type resolve(base_callable *caller, char *data, call_graph &cg, const ndt::type &dst_tp, size_t nsrc,
                        const ndt::type *src_tp, size_t nkwd, const array *kwds,
                        const std::map<std::string, ndt::type> &tp_vars) {
        data_type child_data;
        if (data == nullptr) {
          if (m_child.is_null()) {
            child_data.child = caller;
          } else {
            child_data.child = m_child.get();
          }
          child_data.tracking = m_tracking;
          child_data.ndim = 0;
          data = reinterpret_cast<char *>(&child_data);

          if (m_tracking) {
          }
        } else {
        }

        const ndt::callable_type *child_tp =
            reinterpret_cast<data_type *>(data)->child->get_type().template extended<ndt::callable_type>();

        bool dst_variadic = dst_tp.is_variadic();
        bool all_same = true;
        if (!dst_tp.is_symbolic()) {
          all_same = dst_tp.get_ndim() == child_tp->get_return_type().get_ndim();
        }
        for (size_t i = 0; i < nsrc; ++i) {
          if (src_tp[i].get_ndim() != child_tp->get_pos_type(i).get_ndim()) {
            all_same = false;
            break;
          }
        }

        if (all_same) {
          return reinterpret_cast<data_type *>(data)->child->resolve(
              this, nullptr, cg,
              dst_tp.is_symbolic() ? reinterpret_cast<data_type *>(data)->child->get_return_type() : dst_tp, nsrc,
              src_tp, nkwd, kwds, tp_vars);
        } else {
          reinterpret_cast<data_type *>(data)->ndim += 1;
        }

        // Do a pass through the src types to classify them
        bool src_all_strided = true, src_all_strided_or_var = true;
        for (size_t i = 0; i < nsrc; ++i) {
          intptr_t src_ndim = src_tp[i].get_ndim() - child_tp->get_pos_type(i).get_ndim();
          switch (src_tp[i].get_id()) {
          case fixed_dim_id:
            break;
          case var_dim_id:
            src_all_strided = false;
            break;
          default:
            // If it's a scalar, allow it to broadcast like
            // a strided dimension
            if (src_ndim > 0) {
              src_all_strided_or_var = false;
            }
            break;
          }
        }

        bool var_broadcast = !src_all_strided;
        for (size_t i = 0; i < N; ++i) {
          var_broadcast &= src_tp[i].get_id() == var_dim_id ||
                           (src_tp[i].get_id() == fixed_dim_id &&
                            src_tp[i].extended<ndt::fixed_dim_type>()->get_fixed_dim_size() == 1);
        }

        if ((dst_variadic || dst_tp.get_id() == fixed_dim_id) && src_all_strided) {
          static callable f = make_callable<elwise_callable<fixed_dim_id, fixed_dim_id, N>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else if (((dst_variadic) || dst_tp.get_id() == var_dim_id) && (var_broadcast || src_all_strided)) {
          static callable f = make_callable<elwise_callable<var_dim_id, fixed_dim_id, N>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        } else if (src_all_strided_or_var) {
          static callable f = make_callable<elwise_callable<fixed_dim_id, var_dim_id, N>>();
          return f->resolve(this, data, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
        }

        std::stringstream ss;
        ss << "Cannot process lifted elwise expression from (";
        for (size_t i = 0; i < nsrc; ++i) {
          ss << src_tp[i];
          if (i != nsrc - 1) {
            ss << ", ";
          }
        }
        ss << ") to " << dst_tp;
        throw std::runtime_error(ss.str());
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd

namespace ndt {

  template <size_t N>
  struct traits<nd::functional::elwise_dispatch_callable<N>> {
    static type equivalent(const type &child_tp) {
      const std::vector<ndt::type> &param_types = child_tp.extended<ndt::callable_type>()->get_pos_types();
      std::vector<ndt::type> out_param_types;
      std::string dimsname("Dims");

      for (const ndt::type &t : param_types) {
        out_param_types.push_back(ndt::make_ellipsis_dim(dimsname, t));
      }

      ndt::type kwd_tp = child_tp.extended<ndt::callable_type>()->get_kwd_struct();
      ndt::type ret_tp = child_tp.extended<ndt::callable_type>()->get_return_type();
      ret_tp = ndt::make_ellipsis_dim(dimsname, ret_tp);

      return ndt::callable_type::make(
          ret_tp, ndt::make_type<ndt::tuple_type>(out_param_types.size(), out_param_types.data()), kwd_tp);
    }
  };

} // namespace dynd::ndt
} // namespace dynd
