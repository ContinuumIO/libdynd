//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/apply.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename func_type, typename R, typename A, typename I, typename K, typename J>
      struct construct_then_apply_callable_kernel;

#define CONSTRUCT_THEN_APPLY_CALLABLE_KERNEL(...)                                                                      \
  template <typename func_type, typename R, typename... A, size_t... I, typename... K, size_t... J>                    \
  struct construct_then_apply_callable_kernel<                                                                         \
      func_type, R, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                    \
      index_sequence<J...>> : base_kernel<construct_then_apply_callable_kernel<func_type, R, type_sequence<A...>,      \
                                                                               index_sequence<I...>,                   \
                                                                               type_sequence<K...>,                    \
                                                                               index_sequence<J...>>,                  \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>> {                                  \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    __VA_ARGS__ construct_then_apply_callable_kernel(args_type args, kwds_type DYND_IGNORE_UNUSED(kwds))               \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...)                                                        \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))                                           \
    {                                                                                                                  \
      *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);                                             \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride, char *const *DYND_IGNORE_UNUSED(src_copy),                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      dst += DYND_THREAD_ID(0) * dst_stride;                                                                           \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);                                           \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                                                                      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),                         \
                                char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,                               \
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),            \
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,   \
                                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,  \
                                const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))   \
    {                                                                                                                  \
      construct_then_apply_callable_kernel::make(ckb, kernreq, ckb_offset, args_type(src_tp, src_arrmeta, kwds),       \
                                                 kwds_type(nkwd, kwds));                                               \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <typename func_type, typename... A, size_t... I, typename... K, size_t... J>                                \
  struct construct_then_apply_callable_kernel<                                                                         \
      func_type, void, type_sequence<A...>, index_sequence<I...>, type_sequence<K...>,                                 \
      index_sequence<J...>> : base_kernel<construct_then_apply_callable_kernel<func_type, void, type_sequence<A...>,   \
                                                                               index_sequence<I...>,                   \
                                                                               type_sequence<K...>,                    \
                                                                               index_sequence<J...>>,                  \
                                          sizeof...(A)>,                                                               \
                              apply_args<type_sequence<A...>, index_sequence<I...>> {                                  \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;                                           \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;                                           \
                                                                                                                       \
    func_type func;                                                                                                    \
                                                                                                                       \
    __VA_ARGS__ construct_then_apply_callable_kernel(args_type args, kwds_type DYND_IGNORE_UNUSED(kwds))               \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...)                                                        \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst), char *const *DYND_IGNORE_UNUSED(src))                              \
    {                                                                                                                  \
      func(apply_arg<A, I>::get(src[I])...);                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),                                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),                                                \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride), size_t count)                             \
    {                                                                                                                  \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                                                           \
                                                                                                                       \
      for (size_t j = 0; j != sizeof...(A); ++j) {                                                                     \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];                                                      \
      }                                                                                                                \
                                                                                                                       \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count; i += DYND_THREAD_COUNT(0)) {               \
        func(apply_arg<A, I>::get(src[I])...);                                                                         \
        for (size_t j = 0; j != sizeof...(A); ++j) {                                                                   \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                                                              \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    static intptr_t instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),                         \
                                char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,                               \
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),            \
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,   \
                                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd,  \
                                const nd::array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))   \
    {                                                                                                                  \
      construct_then_apply_callable_kernel::make(ckb, kernreq, ckb_offset, args_type(src_tp, src_arrmeta, kwds),       \
                                                 kwds_type(nkwd, kwds));                                               \
      return ckb_offset;                                                                                               \
    }                                                                                                                  \
  }

      CONSTRUCT_THEN_APPLY_CALLABLE_KERNEL();

#undef CONSTRUCT_THEN_APPLY_CALLABLE_KERNEL

    } // namespace dynd::nd::functional::detail

    template <typename CallableType, typename... K>
    using construct_then_apply_callable_kernel =
        detail::construct_then_apply_callable_kernel<CallableType, typename return_of<CallableType>::type,
                                                     as_apply_arg_sequence<CallableType, arity_of<CallableType>::value>,
                                                     make_index_sequence<arity_of<CallableType>::value>,
                                                     type_sequence<K...>, make_index_sequence<sizeof...(K)>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
