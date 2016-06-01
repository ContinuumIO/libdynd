//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/apply.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename func_type, typename R, typename A, typename I, typename K, typename J>
      struct construct_then_apply_callable_kernel;

#define CONSTRUCT_THEN_APPLY_CALLABLE_KERNEL(...)                                                                      \
  template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>                    \
  struct construct_then_apply_callable_kernel<func_type, R, type_sequence<A...>, std::index_sequence<I...>,            \
                                              type_sequence<K...>, std::index_sequence<J...>>                          \
      : base_strided_kernel<                                                                                           \
            construct_then_apply_callable_kernel<func_type, R, type_sequence<A...>, std::index_sequence<I...>,         \
                                                 type_sequence<K...>, std::index_sequence<J...>>,                      \
            sizeof...(A)>,                                                                                             \
        apply_args<type_sequence<A...>, std::index_sequence<I...>> {                                                   \
    typedef apply_args<type_sequence<A...>, std::index_sequence<I...>> args_type;                                      \
    typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;                                      \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    construct_then_apply_callable_kernel(args_type args, kwds_type DYND_IGNORE_UNUSED(kwds))                           \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...) {}                                                     \
                                                                                                                       \
    void single(char *dst, char *const *DYND_IGNORE_UNUSED(src)) {                                                     \
      *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);                                             \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>                                \
  struct construct_then_apply_callable_kernel<func_type, void, type_sequence<A...>, std::index_sequence<I...>,         \
                                              type_sequence<K...>, std::index_sequence<J...>>                          \
      : base_strided_kernel<                                                                                           \
            construct_then_apply_callable_kernel<func_type, void, type_sequence<A...>, std::index_sequence<I...>,      \
                                                 type_sequence<K...>, std::index_sequence<J...>>,                      \
            sizeof...(A)>,                                                                                             \
        apply_args<type_sequence<A...>, std::index_sequence<I...>> {                                                   \
    typedef apply_args<type_sequence<A...>, std::index_sequence<I...>> args_type;                                      \
    typedef apply_kwds<type_sequence<K...>, std::index_sequence<J...>> kwds_type;                                      \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    construct_then_apply_callable_kernel(args_type args, kwds_type DYND_IGNORE_UNUSED(kwds))                           \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...) {}                                                     \
                                                                                                                       \
    void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src)) {                                        \
      func(apply_arg<A, I>::get(src[I])...);                                                                           \
    }                                                                                                                  \
  }

      CONSTRUCT_THEN_APPLY_CALLABLE_KERNEL();

#undef CONSTRUCT_THEN_APPLY_CALLABLE_KERNEL

    } // namespace dynd::nd::functional::detail

    template <typename func_type, typename... K>
    using construct_then_apply_callable_kernel =
        detail::construct_then_apply_callable_kernel<func_type, typename return_of<func_type>::type,
                                                     as_apply_arg_sequence<func_type, arity_of<func_type>::value>,
                                                     std::make_index_sequence<arity_of<func_type>::value>,
                                                     type_sequence<K...>, std::make_index_sequence<sizeof...(K)>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
