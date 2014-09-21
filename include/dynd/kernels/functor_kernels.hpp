//
// Copyright (C) 2011-14 Mark Wiebe, Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND__KERNELS_FUNCTOR_KERNELS_HPP
#define DYND__KERNELS_FUNCTOR_KERNELS_HPP

#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/pp/meta.hpp>
#include <dynd/types/funcproto_type.hpp>

namespace dynd { namespace nd { namespace detail {

template <typename T>
class typed_param_from_bytes {
public:
    void init(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta)) {
    }

    T &val(char *data) {
        return *reinterpret_cast<T *>(data);
    }

    const T &val(const char *data) {
        return *reinterpret_cast<const T *>(data);
    }
};

template <typename T, int N>
class typed_param_from_bytes<nd::strided_vals<T, N> > {
private:
    nd::strided_vals<T, N> m_strided;

public:
    void init(const ndt::type &DYND_UNUSED(tp), const char *arrmeta) {
        m_strided.init(reinterpret_cast<const size_stride_t *>(arrmeta));
    }

    nd::strided_vals<T, N> &val(char *data) {
        m_strided.set_readonly_originptr(data);
        return m_strided;
    }

    const nd::strided_vals<T, N> &val(const char *data) {
        m_strided.set_readonly_originptr(data);
        return m_strided;
    }
};

#define DECL_TYPED_PARAM_FROM_BYTES(TYPENAME, NAME) DYND_PP_META_DECL(typed_param_from_bytes<TYPENAME>, NAME)
#define INIT_TYPED_PARAM_FROM_BYTES(NAME, TP, ARRMETA) NAME.init(TP, ARRMETA)
#define PARTIAL_DECAY(TYPENAME) std::remove_cv<typename std::remove_reference<TYPENAME>::type>::type
#define PASS(NAME, ARG) NAME.val(ARG)

template <typename func_type, typename funcproto_type, bool aux_buffered, bool thread_aux_buffered>
struct functor_ck;

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value.
 */
#define FUNCTOR_CK(N) \
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, N), false, false> \
      : kernels::expr_ck<functor_ck<func_type, R DYND_PP_META_NAME_RANGE(A, N), false, false>, N> { \
        typedef R (funcproto_type)DYND_PP_META_NAME_RANGE(A, N); \
\
        typedef functor_ck self_type; \
        typedef kernels::expr_ck<self_type, N> parent_type; \
\
        typedef char *dst_bytes_type; \
        typedef typename std::conditional<is_const_funcproto<funcproto_type>::value, const char *, char *>::type src_bytes_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
\
        using parent_type::single; \
\
        inline void single(dst_bytes_type dst, const src_bytes_type *src) { \
            *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N))); \
        } \
\
        using parent_type::strided; \
\
        inline void strided(dst_bytes_type dst, intptr_t dst_stride, \
                            const src_bytes_type *src, const intptr_t *src_stride, \
                            size_t count) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(src_bytes_type, N), DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_AT_RANGE(src, N)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, N), DYND_PP_META_NAME_RANGE(src_stride, N), DYND_PP_META_AT_RANGE(src_stride, N)); \
            for (size_t i = 0; i < count; ++i) { \
                *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
                                    kernel_request_t kernreq, aux_buffer *DYND_UNUSED(aux), \
                                    const eval::eval_context *DYND_UNUSED(ectx)) { \
            bool kernreq_const = (kernreq == kernel_request_const_single) || (kernreq == kernel_request_const_strided); \
            if (kernreq_const != af_self->func_proto.tcast<dynd::funcproto_type>()->get_const()) { \
                std::stringstream ss; \
                ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, kernreq_const) \
                   << " do not match the arrfunc proto " << af_self->func_proto; \
                throw type_error(ss.str()); \
            } \
            for (intptr_t i = 0; i < N; ++i) { \
                if (src_tp[i] != af_self->get_param_type(i)) { \
                    std::stringstream ss; \
                    ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, kernreq_const) \
                       << " do not match the arrfunc proto " << af_self->func_proto; \
                    throw type_error(ss.str()); \
                } \
            } \
            if (dst_tp != af_self->get_return_type()) { \
                std::stringstream ss; \
                ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, kernreq_const) \
                   << " do not match the arrfunc proto " << af_self->func_proto; \
                throw type_error(ss.str()); \
            } \
\
            self_type *e = self_type::create(ckb, kernreq, ckb_offset); \
            e->func = *af_self->get_data_as<func_type>(); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
\
            return ckb_offset; \
        } \
    }; \
\
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N)), typename aux_buffer_type> \
    struct functor_ck<func_type, R DYND_PP_APPEND(aux_buffer_type *, DYND_PP_META_NAME_RANGE(A, N)), true, false> \
      : kernels::expr_ck<functor_ck<func_type, R DYND_PP_APPEND(aux_buffer_type *, DYND_PP_META_NAME_RANGE(A, N)), true, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
        aux_buffer_type *aux; \
\
        inline void single(char *dst, const char *const *src) { \
            *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N)), this->aux); \
        } \
\
        inline void strided(char *dst, intptr_t dst_stride, \
                            const char *const *src, const intptr_t *src_stride, \
                            size_t count) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, N), DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_AT_RANGE(src, N)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, N), DYND_PP_META_NAME_RANGE(src_stride, N), DYND_PP_META_AT_RANGE(src_stride, N)); \
            for (size_t i = 0; i < count; ++i) { \
                *reinterpret_cast<R *>(dst) = this->func(DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N)), this->aux); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
                                    kernel_request_t kernreq, aux_buffer *aux, \
                                    const eval::eval_context *DYND_UNUSED(ectx)) { \
            for (intptr_t i = 0; i < N; ++i) { \
                if (src_tp[i] != af_self->get_param_type(i)) { \
                    std::stringstream ss; \
                    ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, true) \
                       << " do not match the arrfunc proto " << af_self->func_proto; \
                    throw type_error(ss.str()); \
                } \
            } \
            if (dst_tp != af_self->get_return_type()) { \
                std::stringstream ss; \
                ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, true) \
                   << " do not match the arrfunc proto " << af_self->func_proto; \
                throw type_error(ss.str()); \
            } \
\
            self_type *e = self_type::create(ckb, kernreq, ckb_offset); \
            e->func = *af_self->get_data_as<func_type>(); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
            e->aux = reinterpret_cast<aux_buffer_type *>(aux); \
\
            return ckb_offset; \
        } \
    };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef FUNCTOR_CK

/**
 * This generates code to instantiate a ckernel calling a C++ function
 * object which returns a value in the first parameter as an output reference.
 */
#define FUNCTOR_CK(N) \
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct functor_ck<func_type, void DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N)), false, false> \
      : kernels::expr_ck<functor_ck<func_type, void DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N)), false, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DECL_TYPED_PARAM_FROM_BYTES(R, from_dst); \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
\
        inline void single(char *dst, const char *const *src) { \
            this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N))); \
        } \
\
        inline void strided(char *dst, intptr_t dst_stride, \
                            const char *const *src, const intptr_t *src_stride, \
                            size_t count) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, N), DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_AT_RANGE(src, N)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, N), DYND_PP_META_NAME_RANGE(src_stride, N), DYND_PP_META_AT_RANGE(src_stride, N)); \
            for (size_t i = 0; i < count; ++i) { \
                this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N))); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *dst_arrmeta, \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
                                    kernel_request_t kernreq, aux_buffer *DYND_UNUSED(aux), \
                                    const eval::eval_context *DYND_UNUSED(ectx)) { \
            for (intptr_t i = 0; i < N; ++i) { \
                if (src_tp[i] != af_self->get_param_type(i)) { \
                    std::stringstream ss; \
                    ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, true) \
                       << " do not match the arrfunc proto " << af_self->func_proto; \
                    throw type_error(ss.str()); \
                } \
            } \
            if (dst_tp != af_self->get_return_type()) { \
                std::stringstream ss; \
                ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, true) \
                   << " do not match the arrfunc proto " << af_self->func_proto; \
                throw type_error(ss.str()); \
            } \
\
            self_type *e = self_type::create(ckb, kernreq, ckb_offset); \
            e->func = *af_self->get_data_as<func_type>(); \
            INIT_TYPED_PARAM_FROM_BYTES(e->from_dst, dst_tp, dst_arrmeta); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
\
            return ckb_offset; \
        } \
    }; \
\
    template <typename func_type, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N)), typename aux_buffer_type> \
    struct functor_ck<func_type, void DYND_PP_APPEND(aux_buffer_type *, DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N))), true, false> \
      : kernels::expr_ck<functor_ck<func_type, void DYND_PP_APPEND(aux_buffer_type *, DYND_PP_PREPEND(R &, DYND_PP_META_NAME_RANGE(A, N))), true, false>, N> { \
        typedef functor_ck self_type; \
        DYND_PP_JOIN_ELWISE_1(DYND_PP_META_TYPEDEF_TYPENAME, (;), \
            DYND_PP_MAP_1(PARTIAL_DECAY, DYND_PP_META_NAME_RANGE(A, N)), DYND_PP_META_NAME_RANGE(D, N)); \
\
        func_type func; \
        DECL_TYPED_PARAM_FROM_BYTES(R, from_dst); \
        DYND_PP_JOIN_ELWISE_1(DECL_TYPED_PARAM_FROM_BYTES, (;), \
            DYND_PP_META_NAME_RANGE(D, N), DYND_PP_META_NAME_RANGE(from_src, N)); \
        aux_buffer_type *aux; \
\
        inline void single(char *dst, const char *const *src) { \
            this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_AT_RANGE(src, N)), this->aux); \
        } \
\
        inline void strided(char *dst, intptr_t dst_stride, \
                            const char *const *src, const intptr_t *src_stride, \
                            size_t count) { \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(const char *, N), DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_AT_RANGE(src, N)); \
            DYND_PP_JOIN_ELWISE_1(DYND_PP_META_DECL_ASGN, (;), \
                DYND_PP_REPEAT_1(intptr_t, N), DYND_PP_META_NAME_RANGE(src_stride, N), DYND_PP_META_AT_RANGE(src_stride, N)); \
            for (size_t i = 0; i < count; ++i) { \
                this->func(PASS(this->from_dst, dst), DYND_PP_JOIN_ELWISE_1(PASS, (,), \
                    DYND_PP_META_NAME_RANGE(this->from_src, N), DYND_PP_META_NAME_RANGE(src, N)), this->aux); \
                dst += dst_stride; \
                DYND_PP_JOIN_ELWISE_1(DYND_PP_META_ADD_ASGN, (;), \
                    DYND_PP_META_NAME_RANGE(src, N), DYND_PP_META_NAME_RANGE(src_stride, N)); \
            } \
        } \
\
        static intptr_t instantiate(const arrfunc_type_data *af_self, \
                                    dynd::ckernel_builder *ckb, intptr_t ckb_offset, \
                                    const ndt::type &dst_tp, const char *dst_arrmeta, \
                                    const ndt::type *src_tp, const char *const *src_arrmeta, \
                                    kernel_request_t kernreq, aux_buffer *aux, \
                                    const eval::eval_context *DYND_UNUSED(ectx)) { \
            for (intptr_t i = 0; i < N; ++i) { \
                if (src_tp[i] != af_self->get_param_type(i)) { \
                    std::stringstream ss; \
                    ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, true) \
                       << " do not match the arrfunc proto " << af_self->func_proto; \
                    throw type_error(ss.str()); \
                } \
            } \
            if (dst_tp != af_self->get_return_type()) { \
                std::stringstream ss; \
                ss << "Provided types " << ndt::make_funcproto(N, src_tp, dst_tp, true) \
                   << " do not match the arrfunc proto " << af_self->func_proto; \
                throw type_error(ss.str()); \
            } \
\
            self_type *e = self_type::create(ckb, kernreq, ckb_offset); \
            e->func = *af_self->get_data_as<func_type>(); \
            INIT_TYPED_PARAM_FROM_BYTES(e->from_dst, dst_tp, dst_arrmeta); \
            DYND_PP_JOIN_ELWISE_1(INIT_TYPED_PARAM_FROM_BYTES, (;), DYND_PP_META_NAME_RANGE(e->from_src, N), \
                DYND_PP_META_AT_RANGE(src_tp, N), DYND_PP_META_AT_RANGE(src_arrmeta, N)); \
            e->aux = reinterpret_cast<aux_buffer_type *>(aux); \
\
            return ckb_offset; \
        } \
    };

DYND_PP_JOIN_MAP(FUNCTOR_CK, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_SRC_MAX)))

#undef FUNCTOR_CK

#undef DECL_TYPED_PARAM_FROM_BYTES
#undef INIT_TYPED_PARAM_FROM_BYTES
#undef PARTIAL_DECAY
#undef PASS

} // namespace dynd::nd::detail

template <typename func_type, typename funcproto_type>
struct functor_ck : detail::functor_ck<func_type, funcproto_type,
    is_aux_buffered<funcproto_type>::value, is_thread_aux_buffered<funcproto_type>::value> {
};

}} // namespace dynd::nd

#endif
