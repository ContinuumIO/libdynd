//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/expr_kernels.hpp>

namespace dynd {
namespace detail {

  template <typename func_type, typename... B>
  struct funcproto {
    typedef
        typename funcproto<decltype(&func_type::operator()), B...>::type type;
  };

  template <typename R, typename... A, typename... B>
  struct funcproto<R(A...), B...> {
    typedef R(type)(A..., B...);
  };

  template <typename R, typename... A, typename... B>
  struct funcproto<R (*)(A...), B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

  template <typename T, typename R, typename... A, typename... B>
  struct funcproto<R (T::*)(A...), B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };

  template <typename T, typename R, typename... A, typename... B>
  struct funcproto<R (T::*)(A...) const, B...> {
    typedef typename funcproto<R(A...), B...>::type type;
  };
}

template <typename func_type, typename... B>
struct funcproto_of {
  typedef typename detail::funcproto<func_type, B...>::type type;
};

template <typename func_type, typename... B>
struct funcproto_of<func_type *, B...> {
  typedef typename funcproto_of<func_type, B...>::type type;
};

template <typename func_type>
struct return_of {
  typedef typename return_of<typename funcproto_of<func_type>::type>::type type;
};

template <typename R, typename... A>
struct return_of<R(A...)> {
  typedef R type;
};

template <typename func_type>
struct args_of {
  typedef typename args_of<typename funcproto_of<func_type>::type>::type type;
};

template <typename R, typename... A>
struct args_of<R(A...)> {
  typedef type_sequence<A...> type;
};

template <typename func_type>
struct arity_of {
  static const size_t value =
      arity_of<typename funcproto_of<func_type>::type>::value;
};

template <typename R, typename... A>
struct arity_of<R(A...)> {
  static const size_t value = sizeof...(A);
};

namespace nd {
  namespace functional {

    template <typename A, size_t I>
    struct apply_arg {
      typedef
          typename std::remove_cv<typename std::remove_reference<A>::type>::type
              D;

      apply_arg(const ndt::type &DYND_UNUSED(tp),
                const char *DYND_UNUSED(arrmeta),
                const nd::array &DYND_UNUSED(kwds))
      {
      }

      DYND_CUDA_HOST_DEVICE D &get(char *data)
      {
        return *reinterpret_cast<D *>(data);
      }
    };

    template <typename T, size_t I>
    struct apply_arg<const fixed_dim<T> &, I> {
      fixed_dim<T> m_vals;

      apply_arg(const ndt::type &tp, const char *arrmeta,
                const nd::array &DYND_UNUSED(kwds)) : m_vals(tp, arrmeta)
      {
          m_vals.set_data(NULL);
//        m_vals.set_data(NULL, reinterpret_cast<const size_stride_t *>(arrmeta),
  //                      reinterpret_cast<start_stop_t *>(
    //                        kwds.p("start_stop").as<intptr_t>()));
      }

      fixed_dim<T> &get(char *data)
      {
        m_vals.set_data(data);
        return m_vals;
      }
    };

    template <typename func_type,
              int N =
                  args_of<typename funcproto_of<func_type>::type>::type::size>
    using as_apply_arg_sequence = typename to<
        typename args_of<typename funcproto_of<func_type>::type>::type,
        N>::type;

    template <typename A, typename I = make_index_sequence<A::size>>
    struct apply_args;

    template <typename... A, size_t... I>
    struct apply_args<type_sequence<A...>, index_sequence<I...>>
        : apply_arg<A, I>... {
      apply_args(const ndt::type *DYND_IGNORE_UNUSED(src_tp),
                 const char *const *DYND_IGNORE_UNUSED(src_arrmeta),
                 const nd::array &kwds)
          : apply_arg<A, I>(src_tp[I], src_arrmeta[I], kwds)...
      {
      }
    };

    template <typename T, size_t I>
    struct apply_kwd {
      T m_val;

      apply_kwd(nd::array val)
      //        : m_val(val.as<T>())
      {
        if (val.get_type().get_type_id() == pointer_type_id) {
          m_val = val.f("dereference").as<T>();
        }
        else {
          m_val = val.as<T>();
        }
      }

      DYND_CUDA_HOST_DEVICE T get() { return m_val; }
    };

    template <typename K, typename J = make_index_sequence<K::size>>
    struct apply_kwds;

    template <>
    struct apply_kwds<type_sequence<>, index_sequence<>> {
      apply_kwds(const nd::array &DYND_UNUSED(kwds)) {}
    };

    template <typename... K, size_t... J>
    struct apply_kwds<type_sequence<K...>, index_sequence<J...>>
        : apply_kwd<K, J>... {
      apply_kwds(const nd::array &kwds) : apply_kwd<K, J>(kwds.at(J))... {}
    };

    template <typename func_type, int N>
    using as_apply_kwd_sequence = typename from<
        typename args_of<typename funcproto_of<func_type>::type>::type,
        N>::type;

    template <kernel_request_t kernreq, typename func_type, func_type func,
              typename R, typename A, typename I, typename K, typename J>
    struct apply_function_ck;

#define APPLY_FUNCTION_CK(KERNREQ, ...)                                        \
  template <typename func_type, func_type func, typename R, typename... A,     \
            size_t... I, typename... K, size_t... J>                           \
  struct apply_function_ck<KERNREQ, func_type, func, R, type_sequence<A...>,   \
                           index_sequence<I...>, type_sequence<K...>,          \
                           index_sequence<J...>>                               \
      : expr_ck<apply_function_ck<KERNREQ, func_type, func, R,                 \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>,  \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_function_ck self_type;                                       \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    __VA_ARGS__ apply_function_ck(args_type args, kwds_type kwds)              \
        : args_type(args), kwds_type(kwds)                                     \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))   \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);    \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      dst += DYND_THREAD_ID(0) * dst_stride;                                   \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        *reinterpret_cast<R *>(dst) =                                          \
            func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);  \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *DYND_UNUSED(self),                    \
                const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,           \
                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),     \
                const char *DYND_UNUSED(dst_arrmeta),                          \
                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,           \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))   \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_type(src_tp, src_arrmeta, kwds),                  \
                        kwds_type(kwds));                                      \
      return ckb_offset;                                                       \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename func_type, func_type func, typename... A, size_t... I,    \
            typename... K, size_t... J>                                        \
  struct apply_function_ck<KERNREQ, func_type, func, void,                     \
                           type_sequence<A...>, index_sequence<I...>,          \
                           type_sequence<K...>, index_sequence<J...>>          \
      : expr_ck<apply_function_ck<KERNREQ, func_type, func, void,              \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>,  \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_function_ck self_type;                                       \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    __VA_ARGS__ apply_function_ck(args_type args, kwds_type kwds)              \
        : args_type(args), kwds_type(kwds)                                     \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst),                            \
                            char *const *DYND_IGNORE_UNUSED(src))              \
    {                                                                          \
      func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);        \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst),                           \
                             intptr_t DYND_UNUSED(dst_stride),                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *DYND_UNUSED(self),                    \
                const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,           \
                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),     \
                const char *DYND_UNUSED(dst_arrmeta),                          \
                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,           \
                const char *const *src_arrmeta, kernel_request_t kernreq,      \
                const eval::eval_context *DYND_UNUSED(ectx),                   \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))   \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_type(src_tp, src_arrmeta, kwds),                  \
                        kwds_type(kwds));                                      \
      return ckb_offset;                                                       \
    }                                                                          \
  }

    APPLY_FUNCTION_CK(kernel_request_host);

#undef APPLY_FUNCTION_CK

    template <kernel_request_t kernreq, typename func_type, func_type func,
              int N>
    using as_apply_function_ck = apply_function_ck<
        kernreq, func_type, func, typename return_of<func_type>::type,
        as_apply_arg_sequence<func_type, N>, make_index_sequence<N>,
        as_apply_kwd_sequence<func_type, N>,
        make_index_sequence<arity_of<func_type>::value - N>>;

    template <kernel_request_t kernreq, typename T, typename mem_func_type,
              typename R, typename A, typename I, typename K, typename J>
    struct apply_member_function_ck;

#define APPLY_MEMBER_FUNCTION_CK(KERNREQ, ...)                                 \
  template <typename T, typename mem_func_type, typename R, typename... A,     \
            size_t... I, typename... K, size_t... J>                           \
  struct apply_member_function_ck<KERNREQ, T *, mem_func_type, R,              \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>   \
      : expr_ck<apply_member_function_ck<                                      \
                    KERNREQ, T *, mem_func_type, R, type_sequence<A...>,       \
                    index_sequence<I...>, type_sequence<K...>,                 \
                    index_sequence<J...>>,                                     \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_member_function_ck self_type;                                \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
    typedef std::pair<T *, mem_func_type> data_type;                           \
                                                                               \
    T *obj;                                                                    \
    mem_func_type mem_func;                                                    \
                                                                               \
    __VA_ARGS__ apply_member_function_ck(T *obj, mem_func_type mem_func,       \
                                         args_type args, kwds_type kwds)       \
        : args_type(args), kwds_type(kwds), obj(obj), mem_func(mem_func)       \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))   \
    {                                                                          \
      *reinterpret_cast<R *>(dst) = (obj->*mem_func)(                          \
          apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);         \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      dst += DYND_THREAD_ID(0) * dst_stride;                                   \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        *reinterpret_cast<R *>(dst) = (obj->*mem_func)(                        \
            apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);       \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(                                                       \
          ckb, kernreq, ckb_offset, self->get_data_as<data_type>()->first,     \
          dynd::detail::make_value_wrapper(                                    \
              self->get_data_as<data_type>()->second),                         \
          args_type(src_tp, src_arrmeta, kwds), kwds_type(kwds));              \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  };                                                                           \
                                                                               \
  template <typename T, typename mem_func_type, typename... A, size_t... I,    \
            typename... K, size_t... J>                                        \
  struct apply_member_function_ck<KERNREQ, T *, mem_func_type, void,           \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>   \
      : expr_ck<apply_member_function_ck<                                      \
                    KERNREQ, T *, mem_func_type, void, type_sequence<A...>,    \
                    index_sequence<I...>, type_sequence<K...>,                 \
                    index_sequence<J...>>,                                     \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_member_function_ck self_type;                                \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
    typedef std::pair<T *, mem_func_type> data_type;                           \
                                                                               \
    T *obj;                                                                    \
    mem_func_type mem_func;                                                    \
                                                                               \
    __VA_ARGS__ apply_member_function_ck(T *obj, mem_func_type mem_func,       \
                                         args_type args, kwds_type kwds)       \
        : args_type(args), kwds_type(kwds), obj(obj), mem_func(mem_func)       \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst),                            \
                            char *const *DYND_IGNORE_UNUSED(src))              \
    {                                                                          \
      (obj->*mem_func)(apply_arg<A, I>::get(src[I])...,                        \
                       apply_kwd<K, J>::get()...);                             \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst),                           \
                             intptr_t DYND_UNUSED(dst_stride),                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        (obj->*mem_func)(apply_arg<A, I>::get(src[I])...,                      \
                         apply_kwd<K, J>::get()...);                           \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(                                                       \
          ckb, kernreq, ckb_offset, self->get_data_as<data_type>()->first,     \
          dynd::detail::make_value_wrapper(                                    \
              self->get_data_as<data_type>()->second),                         \
          args_type(src_tp, src_arrmeta, kwds), kwds_type(kwds));              \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  }

    APPLY_MEMBER_FUNCTION_CK(kernel_request_host);

    template <typename T, typename mem_func_type, typename R, typename... A,
              size_t... I, typename... K, size_t... J>
    intptr_t
    apply_member_function_ck<kernel_request_host, T *, mem_func_type, R,
                             type_sequence<A...>, index_sequence<I...>,
                             type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

    template <typename T, typename mem_func_type, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t
    apply_member_function_ck<kernel_request_host, T *, mem_func_type, void,
                             type_sequence<A...>, index_sequence<I...>,
                             type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#ifdef __CUDACC__

    APPLY_MEMBER_FUNCTION_CK(kernel_request_cuda_device, __device__);

    template <typename T, typename mem_func_type, typename R, typename... A,
              size_t... I, typename... K, size_t... J>
    intptr_t
    apply_member_function_ck<kernel_request_cuda_device, T *, mem_func_type, R,
                             type_sequence<A...>, index_sequence<I...>,
                             type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return cuda_launch_ck<sizeof...(A)>::template instantiate<
          &instantiate_without_cuda_launch>(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

    template <typename T, typename mem_func_type, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t
    apply_member_function_ck<kernel_request_cuda_device, T *, mem_func_type,
                             void, type_sequence<A...>, index_sequence<I...>,
                             type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return cuda_launch_ck<sizeof...(A)>::template instantiate<
          &instantiate_without_cuda_launch>(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#endif

#undef APPLY_MEMBER_FUNCTION_CK

    template <kernel_request_t kernreq, typename T, typename mem_func_type,
              int N>
    using as_apply_member_function_ck = apply_member_function_ck<
        kernreq, T, mem_func_type, typename return_of<mem_func_type>::type,
        as_apply_arg_sequence<mem_func_type, N>, make_index_sequence<N>,
        as_apply_kwd_sequence<mem_func_type, N>,
        make_index_sequence<arity_of<mem_func_type>::value - N>>;

    template <kernel_request_t kernreq, typename func_type, typename R,
              typename A, typename I, typename K, typename J>
    struct apply_callable_ck;

#define APPLY_CALLABLE_CK(KERNREQ, ...)                                        \
  template <typename func_type, typename R, typename... A, size_t... I,        \
            typename... K, size_t... J>                                        \
  struct apply_callable_ck<KERNREQ, func_type, R, type_sequence<A...>,         \
                           index_sequence<I...>, type_sequence<K...>,          \
                           index_sequence<J...>>                               \
      : expr_ck<apply_callable_ck<KERNREQ, func_type, R, type_sequence<A...>,  \
                                  index_sequence<I...>, type_sequence<K...>,   \
                                  index_sequence<J...>>,                       \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_callable_ck self_type;                                       \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    func_type func;                                                            \
                                                                               \
    __VA_ARGS__ apply_callable_ck(func_type func, args_type args,              \
                                  kwds_type kwds)                              \
        : args_type(args), kwds_type(kwds), func(func)                         \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))   \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);    \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      dst += DYND_THREAD_ID(0) * dst_stride;                                   \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        *reinterpret_cast<R *>(dst) =                                          \
            func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);  \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(                                                       \
          ckb, kernreq, ckb_offset,                                            \
          dynd::detail::make_value_wrapper(*self->get_data_as<func_type>()),   \
          args_type(src_tp, src_arrmeta, kwds), kwds_type(kwds));              \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  };                                                                           \
                                                                               \
  template <typename func_type, typename... A, size_t... I, typename... K,     \
            size_t... J>                                                       \
  struct apply_callable_ck<KERNREQ, func_type, void, type_sequence<A...>,      \
                           index_sequence<I...>, type_sequence<K...>,          \
                           index_sequence<J...>>                               \
      : expr_ck<apply_callable_ck<KERNREQ, func_type, void,                    \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>,  \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_callable_ck self_type;                                       \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    func_type func;                                                            \
                                                                               \
    __VA_ARGS__ apply_callable_ck(func_type func, args_type args,              \
                                  kwds_type kwds)                              \
        : args_type(args), kwds_type(kwds), func(func)                         \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst),                            \
                            char *const *DYND_IGNORE_UNUSED(src))              \
    {                                                                          \
      func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);        \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst),                           \
                             intptr_t DYND_UNUSED(dst_stride),                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        func(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);      \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(                                                       \
          ckb, kernreq, ckb_offset,                                            \
          dynd::detail::make_value_wrapper(*self->get_data_as<func_type>()),   \
          args_type(src_tp, src_arrmeta, kwds), kwds_type(kwds));              \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  }

    APPLY_CALLABLE_CK(kernel_request_host);

    template <typename func_type, typename R, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t apply_callable_ck<kernel_request_host, func_type, R,
                               type_sequence<A...>, index_sequence<I...>,
                               type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#ifdef __CUDACC__

    APPLY_CALLABLE_CK(kernel_request_cuda_device, __device__);

    template <typename func_type, typename R, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t apply_callable_ck<kernel_request_cuda_device, func_type, R,
                               type_sequence<A...>, index_sequence<I...>,
                               type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return cuda_launch_ck<sizeof...(A)>::template instantiate<
          &instantiate_without_cuda_launch>(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

    template <typename func_type, typename... A, size_t... I, typename... K,
              size_t... J>
    intptr_t apply_callable_ck<kernel_request_cuda_device, func_type, void,
                               type_sequence<A...>, index_sequence<I...>,
                               type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return cuda_launch_ck<sizeof...(A)>::template instantiate<
          &instantiate_without_cuda_launch>(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#endif

#undef APPLY_CALLABLE_CK

#define APPLY_CALLABLE_CK(KERNREQ, ...)                                        \
  template <typename func_type, typename R, typename... A, size_t... I,        \
            typename... K, size_t... J>                                        \
  struct apply_callable_ck<KERNREQ, func_type *, R, type_sequence<A...>,       \
                           index_sequence<I...>, type_sequence<K...>,          \
                           index_sequence<J...>>                               \
      : expr_ck<apply_callable_ck<KERNREQ, func_type *, R,                     \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>,  \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_callable_ck self_type;                                       \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    func_type *func;                                                           \
                                                                               \
    __VA_ARGS__ apply_callable_ck(func_type *func, args_type args,             \
                                  kwds_type kwds)                              \
        : args_type(args), kwds_type(kwds), func(func)                         \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))   \
    {                                                                          \
      *reinterpret_cast<R *>(dst) =                                            \
          (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...); \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      dst += DYND_THREAD_ID(0) * dst_stride;                                   \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        *reinterpret_cast<R *>(dst) = (*func)(apply_arg<A, I>::get(src[I])..., \
                                              apply_kwd<K, J>::get()...);      \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
                                                                               \
      self_type::create(                                                       \
          ckb, kernreq, ckb_offset, *self->get_data_as<func_type *>(),         \
          args_type(src_tp, src_arrmeta, kwds), kwds_type(kwds));              \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  };                                                                           \
                                                                               \
  template <typename func_type, typename... A, size_t... I, typename... K,     \
            size_t... J>                                                       \
  struct apply_callable_ck<KERNREQ, func_type *, void, type_sequence<A...>,    \
                           index_sequence<I...>, type_sequence<K...>,          \
                           index_sequence<J...>>                               \
      : expr_ck<apply_callable_ck<KERNREQ, func_type *, void,                  \
                                  type_sequence<A...>, index_sequence<I...>,   \
                                  type_sequence<K...>, index_sequence<J...>>,  \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>>,                 \
        apply_kwds<type_sequence<K...>, index_sequence<J...>> {                \
    typedef apply_callable_ck self_type;                                       \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    func_type *func;                                                           \
                                                                               \
    __VA_ARGS__ apply_callable_ck(func_type *func, args_type args,             \
                                  kwds_type kwds)                              \
        : args_type(args), kwds_type(kwds), func(func)                         \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst),                            \
                            char *const *DYND_IGNORE_UNUSED(src))              \
    {                                                                          \
      (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);     \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst),                           \
                             intptr_t DYND_UNUSED(dst_stride),                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        (*func)(apply_arg<A, I>::get(src[I])..., apply_kwd<K, J>::get()...);   \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *self,                                         \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
                                                                               \
      self_type::create(                                                       \
          ckb, kernreq, ckb_offset, *self->get_data_as<func_type *>(),         \
          args_type(src_tp, src_arrmeta, kwds), kwds_type(kwds));              \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  }

    APPLY_CALLABLE_CK(kernel_request_host);

    template <typename func_type, typename R, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t apply_callable_ck<kernel_request_host, func_type *, R,
                               type_sequence<A...>, index_sequence<I...>,
                               type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

    template <typename func_type, typename... A, size_t... I, typename... K,
              size_t... J>
    intptr_t apply_callable_ck<kernel_request_host, func_type *, void,
                               type_sequence<A...>, index_sequence<I...>,
                               type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#undef APPLY_CALLABLE_CK

    template <kernel_request_t kernreq, typename func_type, int N>
    using as_apply_callable_ck = apply_callable_ck<
        kernreq, func_type, typename return_of<func_type>::type,
        as_apply_arg_sequence<func_type, N>, make_index_sequence<N>,
        as_apply_kwd_sequence<func_type, N>,
        make_index_sequence<arity_of<func_type>::value - N>>;

    template <kernel_request_t kernreq, typename func_type, typename R,
              typename A, typename I, typename K, typename J>
    struct construct_then_apply_callable_ck;

#define CONSTRUCT_THEN_APPLY_CALLABLE_CK(KERNREQ, ...)                         \
  template <typename func_type, typename R, typename... A, size_t... I,        \
            typename... K, size_t... J>                                        \
  struct construct_then_apply_callable_ck<                                     \
      KERNREQ, func_type, R, type_sequence<A...>, index_sequence<I...>,        \
      type_sequence<K...>, index_sequence<J...>>                               \
      : expr_ck<construct_then_apply_callable_ck<                              \
                    KERNREQ, func_type, R, type_sequence<A...>,                \
                    index_sequence<I...>, type_sequence<K...>,                 \
                    index_sequence<J...>>,                                     \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>> {                \
    typedef construct_then_apply_callable_ck self_type;                        \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    func_type func;                                                            \
                                                                               \
    __VA_ARGS__                                                                \
    construct_then_apply_callable_ck(args_type args,                           \
                                     kwds_type DYND_IGNORE_UNUSED(kwds))       \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...)                \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *dst, char *const *DYND_IGNORE_UNUSED(src))   \
    {                                                                          \
      *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);     \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *dst, intptr_t dst_stride,                   \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      dst += DYND_THREAD_ID(0) * dst_stride;                                   \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        *reinterpret_cast<R *>(dst) = func(apply_arg<A, I>::get(src[I])...);   \
        dst += DYND_THREAD_COUNT(0) * dst_stride;                              \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_type(src_tp, src_arrmeta, kwds),                  \
                        kwds_type(kwds));                                      \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  };                                                                           \
                                                                               \
  template <typename func_type, typename... A, size_t... I, typename... K,     \
            size_t... J>                                                       \
  struct construct_then_apply_callable_ck<                                     \
      KERNREQ, func_type, void, type_sequence<A...>, index_sequence<I...>,     \
      type_sequence<K...>, index_sequence<J...>>                               \
      : expr_ck<construct_then_apply_callable_ck<                              \
                    KERNREQ, func_type, void, type_sequence<A...>,             \
                    index_sequence<I...>, type_sequence<K...>,                 \
                    index_sequence<J...>>,                                     \
                KERNREQ, sizeof...(A)>,                                        \
        apply_args<type_sequence<A...>, index_sequence<I...>> {                \
    typedef construct_then_apply_callable_ck self_type;                        \
    typedef apply_args<type_sequence<A...>, index_sequence<I...>> args_type;   \
    typedef apply_kwds<type_sequence<K...>, index_sequence<J...>> kwds_type;   \
                                                                               \
    func_type func;                                                            \
                                                                               \
    __VA_ARGS__                                                                \
    construct_then_apply_callable_ck(args_type args,                           \
                                     kwds_type DYND_IGNORE_UNUSED(kwds))       \
        : args_type(args), func(kwds.apply_kwd<K, J>::get()...)                \
    {                                                                          \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void single(char *DYND_UNUSED(dst),                            \
                            char *const *DYND_IGNORE_UNUSED(src))              \
    {                                                                          \
      func(apply_arg<A, I>::get(src[I])...);                                   \
    }                                                                          \
                                                                               \
    __VA_ARGS__ void strided(char *DYND_UNUSED(dst),                           \
                             intptr_t DYND_UNUSED(dst_stride),                 \
                             char *const *DYND_IGNORE_UNUSED(src_copy),        \
                             const intptr_t *DYND_IGNORE_UNUSED(src_stride),   \
                             size_t count)                                     \
    {                                                                          \
      dynd::detail::array_wrapper<char *, sizeof...(A)> src;                   \
                                                                               \
      for (size_t j = 0; j != sizeof...(A); ++j) {                             \
        src[j] = src_copy[j] + DYND_THREAD_ID(0) * src_stride[j];              \
      }                                                                        \
                                                                               \
      for (std::ptrdiff_t i = DYND_THREAD_ID(0); i < (std::ptrdiff_t)count;    \
           i += DYND_THREAD_COUNT(0)) {                                        \
        func(apply_arg<A, I>::get(src[I])...);                                 \
        for (size_t j = 0; j != sizeof...(A); ++j) {                           \
          src[j] += DYND_THREAD_COUNT(0) * src_stride[j];                      \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    static intptr_t instantiate_without_cuda_launch(                           \
        const arrfunc_type_data *DYND_UNUSED(self),                            \
        const arrfunc_type *DYND_UNUSED(self_tp), void *ckb,                   \
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),             \
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),      \
        const ndt::type *src_tp, const char *const *src_arrmeta,               \
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx), \
        const nd::array &kwds,                                                 \
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))           \
    {                                                                          \
      self_type::create(ckb, kernreq, ckb_offset,                              \
                        args_type(src_tp, src_arrmeta, kwds),                  \
                        kwds_type(kwds));                                      \
      return ckb_offset;                                                       \
    }                                                                          \
                                                                               \
    static intptr_t                                                            \
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,    \
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,       \
                const char *dst_arrmeta, intptr_t nsrc,                        \
                const ndt::type *src_tp, const char *const *src_arrmeta,       \
                kernel_request_t kernreq, const eval::eval_context *ectx,      \
                const nd::array &kwds,                                         \
                const std::map<nd::string, ndt::type> &tp_vars);               \
  }

    CONSTRUCT_THEN_APPLY_CALLABLE_CK(kernel_request_host);

    template <typename func_type, typename R, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t construct_then_apply_callable_ck<
        kernel_request_host, func_type, R, type_sequence<A...>,
        index_sequence<I...>, type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return instantiate_without_cuda_launch(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#ifdef __CUDACC__

    CONSTRUCT_THEN_APPLY_CALLABLE_CK(kernel_request_cuda_device, __device__);

    template <typename func_type, typename R, typename... A, size_t... I,
              typename... K, size_t... J>
    intptr_t construct_then_apply_callable_ck<
        kernel_request_cuda_device, func_type, R, type_sequence<A...>,
        index_sequence<I...>, type_sequence<K...>, index_sequence<J...>>::
        instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, intptr_t nsrc,
                    const ndt::type *src_tp, const char *const *src_arrmeta,
                    kernel_request_t kernreq, const eval::eval_context *ectx,
                    const nd::array &kwds,
                    const std::map<nd::string, ndt::type> &tp_vars)
    {
      return cuda_launch_ck<arity_of<func_type>::value>::template instantiate<
          &instantiate_without_cuda_launch>(
          self, self_tp, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
          src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }

#endif

#undef CONSTRUCT_THEN_APPLY_CALLABLE_CK

    template <kernel_request_t kernreq, typename func_type, typename... K>
    using as_construct_then_apply_callable_ck =
        construct_then_apply_callable_ck<
            kernreq, func_type, typename return_of<func_type>::type,
            as_apply_arg_sequence<func_type, arity_of<func_type>::value>,
            make_index_sequence<arity_of<func_type>::value>,
            type_sequence<K...>, make_index_sequence<sizeof...(K)>>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd