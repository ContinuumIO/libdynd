//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/shape_tools.hpp>
#include <dynd/func/take_by_pointer_arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>

using namespace std;
using namespace dynd;

struct apply_ck : kernels::expr_ck<apply_ck, 2> {
    intptr_t dst_dim_size, src0_dim_size, src1_stride;
    intptr_t dst_stride, src0_stride;

    void single(char *dst, const char *const *src) {
        ckernel_prefix *child = get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        const char *src0 = src[0];
        const char *src1 = src[1];

        for (intptr_t i = 0; i < dst_dim_size; ++i) {
            intptr_t ix = apply_single_index(*reinterpret_cast<const intptr_t *>(src1), src0_dim_size, NULL);
            const char *child_src0 = src0 + ix * src0_stride;
            const char *child_pointer_src0 = reinterpret_cast<const char *>(&child_src0);
            child_fn(dst, &child_pointer_src0, child);
            dst += dst_stride;
            src1 += src1_stride;
        }
    }
};

static intptr_t instantiate_apply(const arrfunc_type_data *DYND_UNUSED(af_self),
                                  dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta,
                                  const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const nd::array &DYND_UNUSED(aux),
                                  const eval::eval_context *ectx)
{
    intptr_t ndim = src_tp[0].get_ndim();

    ndt::type dst_el_tp, src0_el_tp, src1_el_tp;
    const char *dst_el_meta, *src0_el_meta, *src1_el_meta;
    intptr_t src1_dim_size;
    for (intptr_t i = 0; i < ndim; ++i) {
        typedef apply_ck self_type;
        self_type *self = self_type::create(ckb, kernreq, ckb_offset);

        if (!dst_tp.get_as_strided(dst_arrmeta, &self->dst_dim_size,
                                   &self->dst_stride, &dst_el_tp, &dst_el_meta)) {
            stringstream ss;
            ss << "indexed take arrfunc: could not process type " << dst_tp;
            ss << " as a strided dimension";
            throw type_error(ss.str());
        }
        if (!src_tp[0].get_as_strided(src_arrmeta[0], &self->src0_dim_size,
                                      &self->src0_stride, &src0_el_tp,
                                      &src0_el_meta)) {
            stringstream ss;
            ss << "indexed take arrfunc: could not process type " << src_tp[0];
            ss << " as a strided dimension";
            throw type_error(ss.str());
        }
        if (!src_tp[1].get_as_strided(src_arrmeta[1], &src1_dim_size,
                                      &self->src1_stride, &src1_el_tp,
                                      &src1_el_meta)) {
            stringstream ss;
            ss << "take arrfunc: could not process type " << src_tp[1];
            ss << " as a strided dimension";
            throw type_error(ss.str());
        }
    }

    return make_assignment_kernel(ckb, ckb_offset, dst_el_tp, dst_el_meta,
                                  dst_el_tp, dst_el_meta,
                                  kernel_request_single, ectx);
}

/** Prepends "Dims..." to all the types in the proto */
/*
static ndt::type apply_proto(const ndt::type& proto)
{
    const funcproto_type *p = proto.tcast<funcproto_type>();
    const ndt::type *param_types = p->get_param_types_raw();
    intptr_t param_count = p->get_param_count() + 1;
    nd::array out_param_types =
        nd::typed_empty(1, &param_count, ndt::make_strided_of_type());
    nd::string dimsname("Dims");
    ndt::type *pt = reinterpret_cast<ndt::type *>(
        out_param_types.get_readwrite_originptr());
    for (intptr_t i = 0, i_end = p->get_param_count(); i != i_end; ++i) {
        pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
    }
    pt[param_count - 1] = ndt::make_ellipsis_dim(dimsname, ndt::make_type<intptr_t>());
    return ndt::make_funcproto(
        out_param_types,
        ndt::make_ellipsis_dim(dimsname, p->get_return_type()));
}
*/

static int resolve_take_dst_type(const arrfunc_type_data *DYND_UNUSED(af_self),
                                 ndt::type &out_dst_tp, const ndt::type *src_tp,
                                 int DYND_UNUSED(throw_on_error))
{
    std::cout << "resolving dst_type" << std::endl;

    ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
    if (mask_el_tp.get_type_id() == bool_type_id) {
        out_dst_tp = ndt::make_var_dim(
            ndt::make_pointer(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type()));
    } else if (mask_el_tp.get_type_id() ==
               (type_id_t)type_id_of<intptr_t>::value) {
        if (src_tp[1].get_type_id() == var_dim_type_id) {
            out_dst_tp = ndt::make_var_dim(
                ndt::make_pointer(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type()));
        } else {
            out_dst_tp = ndt::make_strided_dim(
                ndt::make_pointer(src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type()));
        }
    } else {
        stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp
           << ", need bool or intptr";
        throw invalid_argument(ss.str());
    }

    return 1;
}


static void resolve_take_dst_shape(const arrfunc_type_data *DYND_UNUSED(af_self),
                                   intptr_t *out_shape, const ndt::type &dst_tp,
                                   const ndt::type *src_tp,
                                   const char *const *src_arrmeta,
                                   const char *const *src_data)
{
    ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
    if (mask_el_tp.get_type_id() == bool_type_id) {
        out_shape[0] = -1;
    } else if (mask_el_tp.get_type_id() ==
               (type_id_t)type_id_of<intptr_t>::value) {
        src_tp[1].extended()->get_shape(1, 0, out_shape, src_arrmeta[1], src_data[1]);
    } else {
        stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp
           << ", need bool or intptr";
        throw invalid_argument(ss.str());
    }
    if (dst_tp.get_ndim() > 1) {
        // If the elements themselves have dimensions, also initialize their
        // shape
        const char *el_arrmeta = src_arrmeta[0];
        ndt::type el_tp = src_tp[0].get_type_at_dimension(
            const_cast<char **>(&el_arrmeta), 1);
        el_tp.extended()->get_shape(dst_tp.get_ndim() - 1, 0, out_shape + 1,
                                    el_arrmeta, NULL);
    }
}

static void free(arrfunc_type_data *) {}

// take_by_pointer ?
void dynd::make_take_by_pointer_arrfunc(arrfunc_type_data *out_af)
{
    static ndt::type param_types[2] = {ndt::type("M * T"), ndt::type("N * Ix")};
    static ndt::type func_proto = ndt::make_funcproto(param_types, ndt::type("R * pointer[T]"));

    out_af->func_proto = func_proto;
    out_af->instantiate = &instantiate_apply;
    out_af->resolve_dst_shape = &resolve_take_dst_shape;
    out_af->resolve_dst_type = &resolve_take_dst_type;
    out_af->free_func = &free;
}

nd::arrfunc dynd::make_take_by_pointer_arrfunc()
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_take_by_pointer_arrfunc(reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}
