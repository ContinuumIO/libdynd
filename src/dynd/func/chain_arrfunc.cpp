//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/chain_arrfunc.hpp>
#include <dynd/buffer_storage.hpp>
#include <dynd/arrmeta_holder.hpp>

using namespace std;
using namespace dynd;

/**
 * A ckernel for chaining two other ckernels, using temporary buffers
 * dynamically allocated on the heap.
 */
struct unary_heap_chain_ck : public kernels::general_ck<unary_heap_chain_ck> {
  // The offset to the second child ckernel
  intptr_t m_second_offset;
  ndt::type m_buf_tp;
  arrmeta_holder m_buf_arrmeta;
  vector<intptr_t> m_buf_shape;

  static void single(char *dst, const char *const *src, ckernel_prefix *rawself)
  {
    self_type *self = get_self(rawself);
    // Allocate a temporary buffer on the heap
    nd::array buf = nd::typed_empty(self->m_buf_shape.size() - 1,
                                    &self->m_buf_shape[0] + 1, self->m_buf_tp);
    char *buf_data = buf.get_readwrite_originptr();
    ckernel_prefix *first = self->get_child_ckernel();
    expr_const_single_t first_fn = first->get_function<expr_const_single_t>();
    ckernel_prefix *second = self->get_child_ckernel(self->m_second_offset);
    expr_const_single_t second_fn = second->get_function<expr_const_single_t>();
    first_fn(buf_data, src, first);
    second_fn(dst, &buf_data, second);
  }

  static void strided(char *dst, intptr_t dst_stride, const char *const *src,
                      const intptr_t *src_stride, size_t count,
                      ckernel_prefix *rawself)
  {
    self_type *self = get_self(rawself);
    // Allocate a temporary buffer on the heap
    nd::array buf = nd::typed_empty(self->m_buf_shape,
                                    ndt::make_strided_dim(self->m_buf_tp));
    char *buf_data = buf.get_readwrite_originptr();
    intptr_t buf_stride = reinterpret_cast<const strided_dim_type_arrmeta *>(
                              buf.get_arrmeta())->stride;
    ckernel_prefix *first = self->get_child_ckernel();
    expr_const_strided_t first_fn = first->get_function<expr_const_strided_t>();
    ckernel_prefix *second = self->get_child_ckernel(self->m_second_offset);
    expr_const_strided_t second_fn = second->get_function<expr_const_strided_t>();
    const char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];

    size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
    first_fn(buf_data, buf_stride, &src0, src_stride, chunk_size, first);
    second_fn(dst, dst_stride, &buf_data, &buf_stride, chunk_size, second);
    count -= chunk_size;
    while (count) {
      src0 += chunk_size * src0_stride;
      dst += chunk_size * dst_stride;
      reset_strided_buffer_array(buf);
      chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
      first_fn(buf_data, buf_stride, &src0, src_stride, chunk_size, first);
      second_fn(dst, dst_stride, &buf_data, &buf_stride, chunk_size, second);
      count -= chunk_size;
    }
  }

  inline void init_kernfunc(kernel_request_t kernreq)
  {
    base.set_expr_function<self_type>(kernreq);
  }

  inline void destruct_children()
  {
    // The first child ckernel
    get_child_ckernel()->destroy();
    // The second child ckernel
    base.destroy_child_ckernel(m_second_offset);
  }
};

struct instantiate_chain_data {
  nd::arrfunc first;
  nd::arrfunc second;
  ndt::type buf_tp;
};

intptr_t dynd::make_chain_buf_tp_ckernel(
    const arrfunc_type_data *first, const arrfunc_type_data *second,
    const ndt::type &buf_tp, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  if (first->get_param_count() == 1) {
    intptr_t root_ckb_offset = ckb_offset;
    unary_heap_chain_ck *self = unary_heap_chain_ck::create(ckb, kernreq, ckb_offset);
    self->m_buf_tp = buf_tp;
    arrmeta_holder(buf_tp).swap(self->m_buf_arrmeta);
    if (buf_tp.get_ndim() == 0 || first->resolve_dst_shape == NULL) {
      self->m_buf_arrmeta.arrmeta_default_construct(0, NULL, true);
      self->m_buf_shape.push_back(DYND_BUFFER_CHUNK_SIZE);
    } else {
      intptr_t ndim = buf_tp.get_ndim();
      vector<intptr_t> shape(ndim + 1);
      shape[0] = DYND_BUFFER_CHUNK_SIZE;
      first->resolve_dst_shape(first, &shape[0] + 1, buf_tp, src_tp,
                               src_arrmeta, NULL);
      self->m_buf_arrmeta.arrmeta_default_construct(ndim, &shape[0] + 1, true);
      self->m_buf_shape.swap(shape);
    }
    ckb_offset = first->instantiate(first, ckb, ckb_offset, buf_tp,
                                    self->m_buf_arrmeta.get(), src_tp,
                                    src_arrmeta, kernreq, NULL, ectx);
    ckb->ensure_capacity(ckb_offset);
    self = ckb->get_at<unary_heap_chain_ck>(root_ckb_offset);
    self->m_second_offset = ckb_offset - root_ckb_offset;
    const char *buf_arrmeta = self->m_buf_arrmeta.get();
    ckb_offset =
        second->instantiate(second, ckb, ckb_offset, dst_tp, dst_arrmeta,
                            &buf_tp, &buf_arrmeta, kernreq, NULL, ectx);
    return ckb_offset;
  } else {
    throw runtime_error("Multi-parameter arrfunc chaining is not implemented");
  }
}

static intptr_t instantiate_chain_buf_tp(
    const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, aux_buffer *DYND_UNUSED(aux), const eval::eval_context *ectx)
{
  const instantiate_chain_data *icd =
      af_self->get_data_as<instantiate_chain_data>();
  return make_chain_buf_tp_ckernel(
      icd->first.get(), icd->second.get(), icd->buf_tp, ckb, ckb_offset, dst_tp,
      dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
}

static void free_chain_arrfunc(arrfunc_type_data *self_af)
{
  self_af->get_data_as<instantiate_chain_data>()->~instantiate_chain_data();
}

void dynd::make_chain_arrfunc(const nd::arrfunc &first,
                               const nd::arrfunc &second,
                               const ndt::type &buf_tp,
                               arrfunc_type_data *out_af)
{
  if (second.get()->func_proto.tcast<funcproto_type>()->get_param_count() !=
      1) {
    stringstream ss;
    ss << "Cannot chain functions " << first << " and " << second
       << ", because the second function is not unary";
    throw invalid_argument(ss.str());
  }
  out_af->free_func = &free_chain_arrfunc;
  out_af->func_proto = ndt::make_funcproto(
      first.get()->func_proto.tcast<funcproto_type>()->get_param_types(),
      second.get()->func_proto.tcast<funcproto_type>()->get_return_type(),
      first.get()->func_proto.tcast<funcproto_type>()->get_const()
          && second.get()->func_proto.tcast<funcproto_type>()->get_const());
  if (buf_tp.get_type_id() == uninitialized_type_id) {
    //out_af->resolve_dst_type = &resolve_chain_dst_type;
    //out_af->resolve_dst_shape = &resolve_chain_dst_shape;
    //out_af->instantiate = &instantiate_chain_resolve;
    throw runtime_error("Chaining functions without a provided intermediate "
                        "type is not implemented");
  } else {
    instantiate_chain_data *icd = out_af->get_data_as<instantiate_chain_data>();
    icd->first = first;
    icd->second = second;
    icd->buf_tp = buf_tp;
    out_af->instantiate = &instantiate_chain_buf_tp;
  }
}

nd::arrfunc dynd::make_chain_arrfunc(const nd::arrfunc &first,
                                     const nd::arrfunc &second,
                                     const ndt::type &buf_tp)
{
  nd::array af = nd::empty(ndt::make_arrfunc());
  make_chain_arrfunc(
      first, second, buf_tp,
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
  af.flag_as_immutable();
  return af;
}
