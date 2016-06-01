//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/index_kernel.hpp>
#include <dynd/type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/index.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  class index_callable : public base_callable {
  public:
    struct data_type {
      intptr_t nindices;
      int *indices;

      data_type(intptr_t nindices, int *indices) : nindices(nindices), indices(indices) {}
      data_type(const array &index) : data_type(index.get_dim_size(), reinterpret_cast<int *>(index.data())) {}

      void next() {
        --nindices;
        ++indices;
      }
    };

    index_callable() : base_callable(ndt::type("(Any, i: Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                      const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return src_tp[0];
    }

    /*
        void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                         const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                         const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                         kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                         const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          ckb->emplace_back<index_kernel<Arg0ID>>(kernreq);
          node = next(node);
          delete reinterpret_cast<data_type *>(data);
        }
    */
  };

  template <>
  class index_callable<ndt::fixed_dim_type> : public base_callable {
  public:
    struct data_type {
      intptr_t nindices;
      int *indices;

      data_type(intptr_t nindices, int *indices) : nindices(nindices), indices(indices) {}
      data_type(const array &index) : data_type(index.get_dim_size(), reinterpret_cast<int *>(index.data())) {}

      void next() {
        --nindices;
        ++indices;
      }
    };

    index_callable() : base_callable(ndt::type("(Any, i: Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      return index->resolve(this, nullptr, cg, dst_tp, nsrc, &child_src_tp, nkwd, kwds, tp_vars);
    }
    /*
        void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                         const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
       *src_arrmeta,
                         kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                         const std::map<std::string, ndt::type> &tp_vars) {
          ckb->emplace_back<index_kernel<fixed_dim_id>>(
              kernreq, *reinterpret_cast<data_type *>(data)->indices,
              reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(src_arrmeta[0])->stride);
          node = next(node);

          reinterpret_cast<data_type *>(data)->next();

          ndt::type child_src_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
          const char *child_src_arrmeta = src_arrmeta[0] + sizeof(ndt::fixed_dim_type::metadata_type);
          index->instantiate(node, data, ckb, dst_tp, dst_arrmeta, nsrc, &child_src_tp, &child_src_arrmeta,
                             kernel_request_single, nkwd, kwds, tp_vars);
        }
    */
  };

} // namespace dynd::nd
} // namespace dynd
