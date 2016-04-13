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
      struct apply_callable_kernel;

      inline size_t select() { return 0; }

      template <typename... ArgTypes>
      size_t &select(iteration_t &arg0, ArgTypes &&... DYND_UNUSED(args)) {
        std::cout << "found iteration_t" << std::endl;
//        arg0.index[arg0.ndim - 1] = 0;
        return arg0.index[arg0.ndim - 1];
      }

      template <typename Arg0Type, typename... ArgTypes>
      decltype(auto) select(Arg0Type &&DYND_UNUSED(arg0), ArgTypes &&... args) {
        return select(std::forward<ArgTypes>(args)...);
      }

      template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_callable_kernel<func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                                   index_sequence<J...>>
          : base_strided_kernel<apply_callable_kernel<func_type, R, type_sequence<A...>, index_sequence<I...>,
                                                      type_sequence<K...>, index_sequence<J...>>,
                                sizeof...(A)>,
            apply_args<type_sequence<A...>, index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;

        func_type func;

        apply_callable_kernel(func_type func, args_type args, kwds_type kwds)
            : args_type(args), kwds_type(kwds), func(func) {}

        void single(char *dst, char *const *DYND_IGNORE_UNUSED(src)) {
          std::cout << "apply_callable_kernel::single" << std::endl;
          *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);
        }

        decltype(auto) begin(char *const *src) {
          std::cout << "apply_callable::begin" << std::endl;
          return select(apply_arg<A, I>::get(src[I])...);
        }
      };

      template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_callable_kernel<func_type, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                                   index_sequence<J...>>
          : base_strided_kernel<apply_callable_kernel<func_type, void, type_sequence<A...>, index_sequence<I...>,
                                                      type_sequence<K...>, index_sequence<J...>>,
                                sizeof...(A)>,
            apply_args<type_sequence<A...>, index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;

        func_type func;

        apply_callable_kernel(func_type func, args_type args, kwds_type kwds)
            : args_type(args), kwds_type(kwds), func(func) {}

        void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src)) {
          func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

      template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_callable_kernel<func_type *, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                                   index_sequence<J...>>
          : base_strided_kernel<apply_callable_kernel<func_type *, R, type_sequence<A...>, index_sequence<I...>,
                                                      type_sequence<K...>, index_sequence<J...>>,
                                sizeof...(A)>,
            apply_args<type_sequence<A...>, index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;

        func_type *func;

        apply_callable_kernel(func_type *func, args_type args, kwds_type kwds)
            : args_type(args), kwds_type(kwds), func(func) {}

        void single(char *dst, char *const *DYND_IGNORE_UNUSED(src)) {
          *reinterpret_cast<R *>(dst) = (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

      template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>
      struct apply_callable_kernel<func_type *, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,
                                   index_sequence<J...>>
          : base_strided_kernel<apply_callable_kernel<func_type *, void, type_sequence<A...>, index_sequence<I...>,
                                                      type_sequence<K...>, index_sequence<J...>>,
                                sizeof...(A)>,
            apply_args<type_sequence<A...>, index_sequence<I...>>,
            apply_kwds<type_sequence<K...>, index_sequence<J...>> {
        typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;
        typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;

        func_type *func;

        apply_callable_kernel(func_type *func, args_type args, kwds_type kwds)
            : args_type(args), kwds_type(kwds), func(func) {}

        void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src)) {
          (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename func_type, int N>
    using apply_callable_kernel = detail::apply_callable_kernel<
        func_type, typename return_of<func_type>::type, as_apply_arg_sequence<func_type, N>, make_index_sequence<N>,
        as_apply_kwd_sequence<func_type, N>, make_index_sequence<arity_of<func_type>::value - N>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
