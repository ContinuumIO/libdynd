//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/base_memory_type.hpp>
#include "assignment_kernels.cpp"

#include "../types/dynd_complex.cu"
#include "../types/dynd_float16.cu"
#include "../types/dynd_float128.cu"
#include "../types/dynd_int128.cu"
#include "../types/dynd_uint128.cu"

#ifdef DYND_CUDA

static void unaligned_copy_single_cuda_host_to_device(char *dst, const char *const *src,
                ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(extra)->data_size;
    throw_if_not_cuda_success(cudaMemcpy(dst, *src, data_size, cudaMemcpyHostToDevice));
}
static void unaligned_copy_strided_cuda_host_to_device(char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(extra)->data_size;

    const char *src0 = *src;
    intptr_t src0_stride = *src_stride;
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src0 += src0_stride) {
        throw_if_not_cuda_success(cudaMemcpy(dst, src0, data_size, cudaMemcpyHostToDevice));
    }
}

static void unaligned_copy_single_cuda_device_to_host(char *dst, const char *const *src,
                ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(extra)->data_size;
    throw_if_not_cuda_success(cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToHost));
}
static void unaligned_copy_strided_cuda_device_to_host(char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(extra)->data_size;

    const char *src0 = *src;
    intptr_t src0_stride = *src_stride;
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src0 += src0_stride) {
        throw_if_not_cuda_success(cudaMemcpy(dst, src0, data_size, cudaMemcpyDeviceToHost));
    }
}

static void unaligned_copy_single_cuda_device_to_device(char *dst, const char *const *src,
                ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(extra)->data_size;
    throw_if_not_cuda_success(cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToDevice));
}
static void unaligned_copy_strided_cuda_device_to_device(char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *extra)
{
    size_t data_size = reinterpret_cast<unaligned_copy_ck *>(extra)->data_size;

    const char *src0 = *src;
    intptr_t src0_stride = *src_stride;
    for (size_t i = 0; i != count; ++i,
                    dst += dst_stride, src0 += src0_stride) {
        throw_if_not_cuda_success(cudaMemcpy(dst, src0, data_size, cudaMemcpyDeviceToDevice));
    }
}

static const ndt::type& get_storage_type(const ndt::type& tp) {
    if (tp.get_kind() == memory_kind) {
        return static_cast<const base_memory_type *>(tp.extended())->get_storage_type();
    } else {
        return tp;
    }
}

size_t dynd::make_cuda_assignment_kernel(
                ckernel_builder *ckb, intptr_t ckb_offset,
                const ndt::type& dst_tp, const char *dst_arrmeta,
                const ndt::type& src_tp, const char *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx)
{
    assign_error_mode errmode;
    if (dst_tp.get_type_id() == cuda_device_type_id && src_tp.get_type_id() == cuda_device_type_id) {
        errmode = ectx->cuda_device_errmode;
    } else {
        errmode = ectx->errmode;
    }

    if (get_storage_type(dst_tp).is_builtin()) {
        if (get_storage_type(src_tp).is_builtin()) {
            if (errmode != assign_error_nocheck && is_lossless_assignment(dst_tp, src_tp)) {
                errmode = assign_error_nocheck;
            }

            if (get_storage_type(dst_tp).extended() == get_storage_type(src_tp).extended()) {
                return make_cuda_pod_typed_data_assignment_kernel(ckb, ckb_offset,
                                dst_tp.get_type_id() == cuda_device_type_id,
                                src_tp.get_type_id() == cuda_device_type_id,
                                dst_tp.get_data_size(), dst_tp.get_data_alignment(),
                                kernreq);
            } else {
                return make_cuda_builtin_type_assignment_kernel(ckb, ckb_offset,
                                dst_tp.get_type_id() == cuda_device_type_id,
                                get_storage_type(dst_tp).get_type_id(),
                                src_tp.get_type_id() == cuda_device_type_id,
                                get_storage_type(src_tp).get_type_id(),
                                kernreq, errmode);
            }
        } else {
            return src_tp.extended()->make_assignment_kernel(ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_tp, src_arrmeta,
                            kernreq, ectx);
        }
    } else {
        return dst_tp.extended()->make_assignment_kernel(ckb, ckb_offset,
                        dst_tp, dst_arrmeta,
                        src_tp, src_arrmeta,
                        kernreq, ectx);
    }
}

template <typename dst_type, typename src_type, assign_error_mode errmode>
struct single_cuda_host_to_device_assigner_builtin {
    static void assign(dst_type *dst, const src_type *src, ckernel_prefix *DYND_UNUSED(extra)) {
        dst_type tmp;
        single_assigner_builtin<dst_type, src_type, errmode>::assign(&tmp, src);
        throw_if_not_cuda_success(cudaMemcpy(dst, &tmp, sizeof(dst_type), cudaMemcpyHostToDevice));
    }
};
template <typename same_type, assign_error_mode errmode>
struct single_cuda_host_to_device_assigner_builtin<same_type, same_type, errmode> {
    static void assign(same_type *dst, const same_type *src, ckernel_prefix *DYND_UNUSED(extra)) {
        throw_if_not_cuda_success(cudaMemcpy(dst, src, sizeof(same_type), cudaMemcpyHostToDevice));
    }
};

namespace {
template<class dst_type, class src_type, assign_error_mode errmode>
struct single_cuda_host_to_device_assigner_as_expr_single {
    static void single(char *dst, const char *const *src,
                       ckernel_prefix *self)
    {
        single_cuda_host_to_device_assigner_builtin<dst_type, src_type, errmode>::assign(
            reinterpret_cast<dst_type *>(dst),
            reinterpret_cast<const src_type *>(*src), self);
    }
};
} // anonymous namespace

static expr_const_single_t assign_table_single_cuda_host_to_device_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            (expr_const_single_t)&single_cuda_host_to_device_assigner_as_expr_single<dst_type, src_type, errmode>::single
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_uint128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef SINGLE_OPERATION_PAIR_LEVEL
};

template <typename dst_type, typename src_type, assign_error_mode errmode>
struct single_cuda_device_to_host_assigner_builtin {
    static void assign(dst_type *dst, const src_type *src, ckernel_prefix *DYND_UNUSED(extra)) {
        src_type tmp;
        throw_if_not_cuda_success(cudaMemcpy(&tmp, src, sizeof(src_type), cudaMemcpyDeviceToHost));
        single_assigner_builtin<dst_type, src_type, errmode>::assign(dst, &tmp);
    }
};
template <typename same_type, assign_error_mode errmode>
struct single_cuda_device_to_host_assigner_builtin<same_type, same_type, errmode> {
    static void assign(same_type *dst, const same_type *src, ckernel_prefix *DYND_UNUSED(extra)) {
        throw_if_not_cuda_success(cudaMemcpy(dst, src, sizeof(same_type), cudaMemcpyDeviceToHost));
    }
};

namespace {
template<class dst_type, class src_type, assign_error_mode errmode>
struct single_cuda_device_to_host_assigner_as_expr_single {
    static void single(char *dst, const char *const *src,
                       ckernel_prefix *self)
    {
        single_cuda_device_to_host_assigner_builtin<dst_type, src_type, errmode>::assign(
            reinterpret_cast<dst_type *>(dst),
            reinterpret_cast<const src_type *>(*src), self);
    }
};
} // anonymous namespace

static expr_const_single_t assign_table_single_cuda_device_to_host_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            (expr_const_single_t)&single_cuda_device_to_host_assigner_as_expr_single<dst_type, src_type, errmode>::single
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_uint128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef SINGLE_OPERATION_PAIR_LEVEL
};

template <typename dst_type, typename src_type, assign_error_mode errmode>
DYND_CUDA_GLOBAL void single_cuda_global_assign_builtin(dst_type *dst, const src_type *src, ckernel_prefix *extra) {
    single_assigner_builtin<dst_type, src_type, errmode>::assign(dst, src);
}

template <typename dst_type, typename src_type, assign_error_mode errmode>
struct single_cuda_device_to_device_assigner_builtin {
    static void assign(dst_type *DYND_UNUSED(dst), const src_type *DYND_UNUSED(src), ckernel_prefix *DYND_UNUSED(extra)) {
        std::stringstream ss;
        ss << "assignment from " << ndt::make_type<src_type>() << " in CUDA global memory to ";
        ss << ndt::make_type<dst_type>() << " in CUDA global memory ";
        ss << "with error mode " << errmode << " is not implemented";
        throw std::runtime_error(ss.str());
    }
};
template <typename dst_type, typename src_type>
struct single_cuda_device_to_device_assigner_builtin<dst_type, src_type, assign_error_nocheck> {
    static void assign(dst_type *dst, const src_type *src, ckernel_prefix *DYND_UNUSED(extra)) {
        single_cuda_global_assign_builtin<dst_type, src_type, assign_error_nocheck><<<1, 1>>>(dst, src, NULL);
        throw_if_not_cuda_success(cudaDeviceSynchronize());
    }
};

namespace {
template<class dst_type, class src_type, assign_error_mode errmode>
struct single_cuda_device_to_device_assigner_as_expr_single {
    static void single(char *dst, const char *const *src,
                       ckernel_prefix *self)
    {
        single_cuda_device_to_device_assigner_builtin<dst_type, src_type, errmode>::assign(
            reinterpret_cast<dst_type *>(dst),
            reinterpret_cast<const src_type *>(*src), self);
    }
};
} // anonymous namespace

static expr_const_single_t assign_table_single_cuda_device_to_device_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            (expr_const_single_t)&single_cuda_device_to_device_assigner_as_expr_single<dst_type, src_type, errmode>::single
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        SINGLE_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_uint128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef SINGLE_OPERATION_PAIR_LEVEL
};

namespace {
    template <typename dst_type, typename src_type, assign_error_mode errmode>
    DYND_CUDA_GLOBAL void strided_cuda_global_assign_builtin(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *DYND_UNUSED(extra))
    {
        unsigned int thread = get_cuda_global_thread<1, 1>();

        if (thread < count) {
            single_assigner_builtin<dst_type, src_type, errmode>::assign(
                                reinterpret_cast<dst_type *>(dst + thread * dst_stride),
                                reinterpret_cast<const src_type *>(src + thread * src_stride));
        }
    }
    template <typename dst_type, typename src_type, assign_error_mode errmode>
    DYND_CUDA_GLOBAL void strided_cuda_global_multiple_assign_builtin(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count_div_threads, size_t count_mod_threads, ckernel_prefix *extra)
    {
        unsigned int thread = get_cuda_global_thread<1, 1>();

/*
        if (thread < count_mod_threads) {
            multiple_assignment_builtin<dst_type, src_type, errmode>::strided_assign(
                            dst + thread * (count_div_threads + 1) * dst_stride, dst_stride,
                            src + thread * (count_div_threads + 1) * src_stride, src_stride,
                            count_div_threads + 1, extra);
        } else {
            multiple_assignment_builtin<dst_type, src_type, errmode>::strided_assign(
                            dst + (thread * count_div_threads + count_mod_threads) * dst_stride, dst_stride,
                            src + (thread * count_div_threads + count_mod_threads) * src_stride, src_stride,
                            count_div_threads, extra);
        }
*/
    }

    template <typename dst_type, typename src_type, assign_error_mode errmode>
    struct multiple_cuda_device_to_device_assignment_builtin {
        static void strided_assign(
                        char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),
                        const char *const *DYND_UNUSED(src), const intptr_t *DYND_UNUSED(src_stride),
                        size_t DYND_UNUSED(count), ckernel_prefix *DYND_UNUSED(extra))
        {
            std::stringstream ss;
            ss << "assignment from " << ndt::make_type<src_type>() << " in CUDA global memory to ";
            ss << ndt::make_type<dst_type>() << " in CUDA global memory ";
            ss << "with error mode " << errmode << " is not implemented";
            throw std::runtime_error(ss.str());
        }
    };
    template <typename dst_type, typename src_type>
    struct multiple_cuda_device_to_device_assignment_builtin<dst_type, src_type, assign_error_nocheck> {
        static void strided_assign(
                        char *dst, intptr_t dst_stride,
                        const char *const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *DYND_UNUSED(extra))
        {
            cuda_global_config<1, 1> config = make_cuda_global_config<1, 1>(count);

            const char *src0 = *src;
            intptr_t src0_stride = *src_stride;
            if (count < config.threads) {
                strided_cuda_global_assign_builtin<dst_type, src_type, assign_error_nocheck>
                                <<<config.grid, config.block>>>(dst, dst_stride, src0, src0_stride,
                                count, NULL);
            } else {
                strided_cuda_global_multiple_assign_builtin<dst_type, src_type, assign_error_nocheck>
                                <<<config.grid, config.block>>>(dst, dst_stride, src0, src0_stride,
                                count / config.threads, count % config.threads, NULL);
            }
            throw_if_not_cuda_success(cudaDeviceSynchronize());
        }
    };
} // anonymous namespace

static expr_const_strided_t assign_table_strided_cuda_device_to_device_kernel[builtin_type_id_count-2][builtin_type_id_count-2][4] =
{
#define STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, errmode) \
            &multiple_cuda_device_to_device_assignment_builtin<dst_type, src_type, errmode>::strided_assign
        
#define ERROR_MODE_LEVEL(dst_type, src_type) { \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_nocheck), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_overflow), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_fractional), \
        STRIDED_OPERATION_PAIR_LEVEL(dst_type, src_type, assign_error_inexact) \
    }

#define SRC_TYPE_LEVEL(dst_type) { \
        ERROR_MODE_LEVEL(dst_type, dynd_bool), \
        ERROR_MODE_LEVEL(dst_type, int8_t), \
        ERROR_MODE_LEVEL(dst_type, int16_t), \
        ERROR_MODE_LEVEL(dst_type, int32_t), \
        ERROR_MODE_LEVEL(dst_type, int64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, uint8_t), \
        ERROR_MODE_LEVEL(dst_type, uint16_t), \
        ERROR_MODE_LEVEL(dst_type, uint32_t), \
        ERROR_MODE_LEVEL(dst_type, uint64_t), \
        ERROR_MODE_LEVEL(dst_type, dynd_int128), \
        ERROR_MODE_LEVEL(dst_type, dynd_float16), \
        ERROR_MODE_LEVEL(dst_type, float), \
        ERROR_MODE_LEVEL(dst_type, double), \
        ERROR_MODE_LEVEL(dst_type, dynd_float128), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<float>), \
        ERROR_MODE_LEVEL(dst_type, dynd_complex<double>) \
    }

    SRC_TYPE_LEVEL(dynd_bool),
    SRC_TYPE_LEVEL(int8_t),
    SRC_TYPE_LEVEL(int16_t),
    SRC_TYPE_LEVEL(int32_t),
    SRC_TYPE_LEVEL(int64_t),
    SRC_TYPE_LEVEL(dynd_int128),
    SRC_TYPE_LEVEL(uint8_t),
    SRC_TYPE_LEVEL(uint16_t),
    SRC_TYPE_LEVEL(uint32_t),
    SRC_TYPE_LEVEL(uint64_t),
    SRC_TYPE_LEVEL(dynd_uint128),
    SRC_TYPE_LEVEL(dynd_float16),
    SRC_TYPE_LEVEL(float),
    SRC_TYPE_LEVEL(double),
    SRC_TYPE_LEVEL(dynd_float128),
    SRC_TYPE_LEVEL(dynd_complex<float>),
    SRC_TYPE_LEVEL(dynd_complex<double>)
#undef SRC_TYPE_LEVEL
#undef ERROR_MODE_LEVEL
#undef STRIDED_OPERATION_PAIR_LEVEL
};

// This is meant to reflect make_builtin_type_assignment_kernel
size_t dynd::make_cuda_builtin_type_assignment_kernel(
                ckernel_builder *out, intptr_t offset_out,
                bool dst_device, type_id_t dst_type_id,
                bool src_device, type_id_t src_type_id,
                kernel_request_t kernreq, assign_error_mode errmode)
{
    if (dst_type_id >= bool_type_id && dst_type_id <= complex_float64_type_id &&
                    src_type_id >= bool_type_id && src_type_id <= complex_float64_type_id &&
                    errmode != assign_error_default) {
        ckernel_prefix *result = out->get_at<ckernel_prefix>(offset_out);
        switch (kernreq) {
            case kernel_request_const_single:
                if (dst_device) {
                    if (src_device) {
                        result->set_function<expr_const_single_t>(
                                        assign_table_single_cuda_device_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    } else {
                        result->set_function<expr_const_single_t>(
                                        assign_table_single_cuda_host_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    }
                } else {
                    if (src_device) {
                        result->set_function<expr_const_single_t>(
                                        assign_table_single_cuda_device_to_host_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    } else {
                        result->set_function<expr_const_single_t>(
                                        assign_table_single_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    }
                }
                break;
            case kernel_request_const_strided:
                if (dst_device) {
                    if (src_device) {
                        result->set_function<expr_const_strided_t>(
                                        assign_table_strided_cuda_device_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    } else {
                        offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, 1, kernreq);
                        result = out->get_at<ckernel_prefix>(offset_out);
                        result->set_function<expr_const_single_t>(
                                        assign_table_single_cuda_host_to_device_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    }
                } else {
                    if (src_device) {
                        offset_out = make_kernreq_to_single_kernel_adapter(out, offset_out, 1, kernreq);
                        result = out->get_at<ckernel_prefix>(offset_out);
                        result->set_function<expr_const_single_t>(
                                        assign_table_single_cuda_device_to_host_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    } else {
                        result->set_function<expr_const_strided_t>(
                                        assign_table_strided_kernel[dst_type_id-bool_type_id]
                                                        [src_type_id-bool_type_id][errmode]);
                    }
                }
                break;
            default: {
                stringstream ss;
                ss << "make_cuda_builtin_type_assignment_function: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }   
        }
        return offset_out + sizeof(ckernel_prefix);
    } else {
        stringstream ss;
        ss << "Cannot assign from " << ndt::type(src_type_id);
        if (src_device) {
            ss << " in CUDA global memory";
        }
        ss << " to " << ndt::type(dst_type_id);
        if (dst_device) {
            ss << " in CUDA global memory";
        }
        throw runtime_error(ss.str());
    }
}

// This is meant to reflect make_pod_typed_data_assignment_kernel
size_t dynd::make_cuda_pod_typed_data_assignment_kernel(
                ckernel_builder *out, intptr_t offset_out,
                bool dst_device, bool src_device,
                size_t data_size, size_t data_alignment,
                kernel_request_t kernreq)
{
    bool single = (kernreq == kernel_request_const_single);
    if (!single && kernreq != kernel_request_const_strided) {
        stringstream ss;
        ss << "make_cuda_pod_typed_data_assignment_kernel: unrecognized request " << (int)kernreq;
        throw runtime_error(ss.str());
    }

    if (dst_device) {
        if (src_device) {
            out->ensure_capacity_leaf(offset_out + sizeof(unaligned_copy_ck));
            ckernel_prefix *result = out->get_at<ckernel_prefix>(offset_out);
            if (single) {
                result->set_function<expr_const_single_t>(&unaligned_copy_single_cuda_device_to_device);
            } else {
                result->set_function<expr_const_strided_t>(&unaligned_copy_strided_cuda_device_to_device);
            }
            reinterpret_cast<unaligned_copy_ck *>(result)->data_size = data_size;
            return offset_out + sizeof(unaligned_copy_ck);
        } else {
            out->ensure_capacity_leaf(offset_out + sizeof(unaligned_copy_ck));
            ckernel_prefix *result = out->get_at<ckernel_prefix>(offset_out);
            if (single) {
                result->set_function<expr_const_single_t>(&unaligned_copy_single_cuda_host_to_device);
            } else {
                result->set_function<expr_const_strided_t>(&unaligned_copy_strided_cuda_host_to_device);
            }
            reinterpret_cast<unaligned_copy_ck *>(result)->data_size = data_size;
            return offset_out + sizeof(unaligned_copy_ck);
        }
    } else {
        if (src_device) {
            out->ensure_capacity_leaf(offset_out + sizeof(unaligned_copy_ck));
            ckernel_prefix *result = out->get_at<ckernel_prefix>(offset_out);
            if (single) {
                result->set_function<expr_const_single_t>(&unaligned_copy_single_cuda_device_to_host);
            } else {
                result->set_function<expr_const_strided_t>(&unaligned_copy_strided_cuda_device_to_host);
            }
            reinterpret_cast<unaligned_copy_ck *>(result)->data_size = data_size;
            return offset_out + sizeof(unaligned_copy_ck);
        } else {
            return make_pod_typed_data_assignment_kernel(out, offset_out, data_size, data_alignment, kernreq);
        }
    }
}

#endif // DYND_CUDA
