//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FOREACH_HPP_
#define _DYND__FOREACH_HPP_

#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/shape_tools.hpp>

namespace dynd { namespace nd {

namespace detail {

template<class FuncProto>
struct foreach_ckernel_instantiator;

template<typename R, typename T0, typename T1>
struct foreach_ckernel_instantiator<R (*)(T0, T1)> {
    typedef foreach_ckernel_instantiator extra_type;

    ckernel_prefix base;
    R (*func)(T0, T1);

    static void single(char *dst, const char * const *src,
                       ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        *reinterpret_cast<R *>(dst) = e->func(
                            *reinterpret_cast<const T0 *>(src[0]),
                            *reinterpret_cast<const T1 *>(src[1]));
    }

    static void strided(char *dst, intptr_t dst_stride,
                        const char * const *src, const intptr_t *src_stride,
                        size_t count, ckernel_prefix *ckp)
    {
        extra_type *e = reinterpret_cast<extra_type *>(ckp);
        R (*func)(T0, T1);
        func = e->func;
        const char *src0 = src[0], *src1 = src[1];
        intptr_t src0_stride = src_stride[0], src1_stride = src_stride[1];
        for (size_t i = 0; i < count; ++i) {
            *reinterpret_cast<R *>(dst) = func(
                                *reinterpret_cast<const T0 *>(src0),
                                *reinterpret_cast<const T1 *>(src1));
            dst += dst_stride;
            src0 += src0_stride;
            src1 += src1_stride;
        }
    }

    static intptr_t instantiate(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
    {
        extra_type *e = out_ckb->get_at<extra_type>(ckb_offset);
        if (kerntype == kernel_request_single) {
            e->base.set_function<expr_single_operation_t>(&extra_type::single);
        } else if (kerntype == kernel_request_strided) {
            e->base.set_function<expr_strided_operation_t>(&extra_type::strided);
        } else {
            throw runtime_error("unsupported kernel request in foreach");
        }
        e->func = reinterpret_cast<R (*)(T0, T1)>(self_data_ptr);
        // No need for a destructor function in this ckernel

        return ckb_offset + sizeof(extra_type);
    }
};

} // namespace detail


template<typename R, typename T0, typename T1>
inline nd::array foreach(const nd::array& a, const nd::array& b, R (*func)(T0, T1))
{
    // No casting for now
    if (a.get_dtype() != ndt::make_type<T0>()) {
        stringstream ss;
        ss << "initial prototype of foreach doesn't implicitly cast ";
        ss << a.get_dtype() << " to " << ndt::make_type<T0>();
        throw type_error(ss.str());
    }
    if (b.get_dtype() != ndt::make_type<T1>()) {
        stringstream ss;
        ss << "initial prototype of foreach doesn't implicitly cast ";
        ss << b.get_dtype() << " to " << ndt::make_type<T1>();
        throw type_error(ss.str());
    }

    // Create a static ckernel_deferred out of the function
    ckernel_deferred ckd;
    ckd.ckernel_funcproto = expr_operation_funcproto;
    ckd.data_types_size = 3;
    ndt::type data_dynd_types[3] = {ndt::make_type<R>(), ndt::make_type<T0>(), ndt::make_type<T1>()};
    ckd.data_dynd_types = data_dynd_types;
    ckd.data_ptr = reinterpret_cast<void *>(func);
    ckd.instantiate_func = &detail::foreach_ckernel_instantiator<R (*)(T0, T1)>::instantiate;
    ckd.free_func = NULL;

    // Get the broadcasted shape
    // TODO: This was hastily grabbed from arithmetic_op.cpp, should be encapsulated much better
    size_t ndim = max(a.get_ndim(), b.get_ndim());
    dimvector result_shape(ndim), tmp_shape(ndim);
    for (size_t j = 0; j != ndim; ++j) {
        result_shape[j] = 1;
    }
    size_t ndim_a = a.get_ndim();
    if (ndim_a > 0) {
        a.get_shape(tmp_shape.get());
        incremental_broadcast(ndim, result_shape.get(), ndim_a, tmp_shape.get());
    }
    size_t ndim_b = b.get_ndim();
    if (ndim_b > 0) {
        b.get_shape(tmp_shape.get());
        incremental_broadcast(ndim, result_shape.get(), ndim_b, tmp_shape.get());
    }

    // Allocate the output array
    nd::array result = nd::make_strided_array(ndt::make_type<R>(), ndim, result_shape.get());

    // Lift the ckernel_deferred to a ckernel which handles the dimensions
    ckernel_builder ckb;
    ndt::type lifted_types[3] = {result.get_type(), a.get_type(), b.get_type()};
    const char *dynd_metadata[3] = {result.get_ndo_meta(), a.get_ndo_meta(), b.get_ndo_meta()};
    make_lifted_expr_ckernel(&ckd, &ckb, 0,
                        lifted_types, dynd_metadata, kernel_request_single);

    // Call the ckernel to do the operation
    ckernel_prefix *ckprefix = ckb.get();
    expr_single_operation_t op = ckprefix->get_function<expr_single_operation_t>();
    const char *src[2] = {a.get_readonly_originptr(), b.get_readonly_originptr()};
    op(result.get_readwrite_originptr(), src, ckprefix);

    return result;
}

}} // namespace dynd::nd

#endif // _DYND__FOREACH_HPP_
