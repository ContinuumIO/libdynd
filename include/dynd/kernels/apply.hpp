//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/iteration_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename A, size_t I>
    struct apply_arg {
      typedef typename std::remove_cv<typename std::remove_reference<A>::type>::type D;

      apply_arg(const char *DYND_UNUSED(arrmeta), const nd::array *DYND_UNUSED(kwds)) {}

      D &get(char *data) { return *reinterpret_cast<D *>(data); }

      char *next(char *data) { return data; }
    };

    template <size_t I>
    struct apply_arg<iteration_t, I> {
      apply_arg(const char *DYND_UNUSED(metadata), const array *DYND_UNUSED(kwds)) {}

      iteration_t &get(char *arg) { return *reinterpret_cast<iteration_t *>(arg); }

      char *next(char *arg) {
//        ++reinterpret_cast<iteration_t *>(arg)->index[0];
        return arg;
      }
    };

    template <typename ElementType, size_t I>
    struct apply_arg<fixed_dim<ElementType>, I> {
      fixed_dim<ElementType> value;

      apply_arg(const char *arrmeta, const nd::array *DYND_UNUSED(kwds)) : value(arrmeta, NULL) {}

      fixed_dim<ElementType> &get(char *data) {
        value.set_data(data);
        return value;
      }

      char *next(char *data) { return data; }
    };

    template <typename func_type, int N = args_of<typename funcproto_of<func_type>::type>::type::size>
    using as_apply_arg_sequence = typename to<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

    template <typename A, typename I = make_index_sequence<A::size>>
    struct apply_args;

    template <typename... A, size_t... I>
    struct apply_args<type_sequence<A...>, index_sequence<I...>> : apply_arg<A, I>... {
      apply_args(const char *const *DYND_IGNORE_UNUSED(src_arrmeta), const nd::array *DYND_IGNORE_UNUSED(kwds))
          : apply_arg<A, I>(src_arrmeta[I], kwds)... {}
    };

    template <typename T, size_t I>
    struct apply_kwd {
      T m_val;

      apply_kwd(nd::array val) : m_val(val.as<T>()) {}

      T get() { return m_val; }
    };

    template <typename K, typename J = make_index_sequence<K::size>>
    struct apply_kwds;

    template <>
    struct apply_kwds<type_sequence<>, index_sequence<>> {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds)) {}
    };

    template <typename... K, size_t... J>
    struct apply_kwds<type_sequence<K...>, index_sequence<J...>> : apply_kwd<K, J>... {
      apply_kwds(intptr_t DYND_UNUSED(nkwd), const nd::array *kwds) : apply_kwd<K, J>(kwds[J])... {}
    };

    template <typename func_type, int N>
    using as_apply_kwd_sequence =
        typename from<typename args_of<typename funcproto_of<func_type>::type>::type, N>::type;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd

#include <dynd/kernels/apply_function_kernel.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>
#include <dynd/kernels/apply_callable_kernel.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>
