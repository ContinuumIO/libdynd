//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/buffered_kernels.hpp>
#include <dynd/buffer_storage.hpp>

using namespace std;
using namespace dynd;

namespace {

struct buffered_ck : nd::base_kernel<buffered_ck> {
  typedef buffered_ck self_type;
  intptr_t m_nsrc;
  vector<intptr_t> m_src_buf_ck_offsets;
  vector<buffer_storage> m_bufs;

  void single(char *dst, char *const *src)
  {
    vector<char *> buf_src(m_nsrc);
    for (intptr_t i = 0; i < m_nsrc; ++i) {
      if (!m_bufs[i].is_null()) {
        m_bufs[i].reset_arrmeta();
        ckernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
        ck->single(m_bufs[i].get_storage(), &src[i]);
        buf_src[i] = m_bufs[i].get_storage();
      }
      else {
        buf_src[i] = src[i];
      }
    }
    ckernel_prefix *child = get_child();
    child->single(dst, &buf_src[0]);
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
  {
    vector<char *> buf_src(m_nsrc);
    vector<intptr_t> buf_stride(m_nsrc);
    ckernel_prefix *child = get_child();
    expr_strided_t child_fn = child->get_function<expr_strided_t>();

    for (intptr_t i = 0; i < m_nsrc; ++i) {
      if (!m_bufs[i].is_null()) {
        buf_src[i] = m_bufs[i].get_storage();
        buf_stride[i] = m_bufs[i].get_stride();
      }
      else {
        buf_src[i] = src[i];
        buf_stride[i] = src_stride[i];
      }
    }

    while (count > 0) {
      size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
      for (intptr_t i = 0; i < m_nsrc; ++i) {
        if (!m_bufs[i].is_null()) {
          m_bufs[i].reset_arrmeta();
          ckernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
          expr_strided_t ck_fn = ck->get_function<expr_strided_t>();
          ck_fn(ck, m_bufs[i].get_storage(), m_bufs[i].get_stride(), &src[i], &src_stride[i], chunk_size);
        }
      }
      child_fn(child, dst, dst_stride, &buf_src[0], &buf_stride[0], chunk_size);
      for (intptr_t i = 0; i < m_nsrc; ++i) {
        if (!m_bufs[i].is_null()) {
          m_bufs[i].reset_arrmeta();
          ckernel_prefix *ck = get_child(m_src_buf_ck_offsets[i]);
          expr_strided_t ck_fn = ck->get_function<expr_strided_t>();
          ck_fn(ck, m_bufs[i].get_storage(), buf_stride[i], &src[i], &src_stride[i], chunk_size);
        }
        else {
          buf_src[i] += chunk_size * buf_stride[i];
        }
      }
      count -= chunk_size;
    }
  }
};

} // anonymous namespace

size_t dynd::make_buffered_ckernel(const nd::base_callable *af, const ndt::callable_type *DYND_UNUSED(af_tp), void *ckb,
                                   intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                   const ndt::type *src_tp, const ndt::type *src_tp_for_af,
                                   const char *const *src_arrmeta, kernel_request_t kernreq,
                                   const eval::eval_context *ectx)
{
  typedef buffered_ck self_type;
  intptr_t root_ckb_offset = ckb_offset;
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  // Prepare the type and buffer info the ckernel needs
  self->m_nsrc = nsrc;
  self->m_bufs.resize(nsrc);
  self->m_src_buf_ck_offsets.resize(nsrc);
  vector<const char *> buffered_arrmeta(nsrc);
  for (intptr_t i = 0; i < nsrc; ++i) {
    if (src_tp[i] == src_tp_for_af[i]) {
      buffered_arrmeta[i] = src_arrmeta[i];
    }
    else {
      self->m_bufs[i].allocate(src_tp_for_af[i]);
      buffered_arrmeta[i] = self->m_bufs[i].get_arrmeta();
    }
  }
  // Instantiate the callable being buffered
  ckb_offset =
      af->instantiate(const_cast<char *>(af->static_data()), NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                      src_tp_for_af, &buffered_arrmeta[0], kernreq, NULL, ectx, 0, NULL, std::map<std::string, ndt::type>());
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
  self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<self_type>(root_ckb_offset);
  // Instantiate assignments for all the buffered operands
  for (intptr_t i = 0; i < nsrc; ++i) {
    if (!self->m_bufs[i].is_null()) {
      self->m_src_buf_ck_offsets[i] = ckb_offset - root_ckb_offset;
      ckb_offset = make_assignment_kernel(ckb, ckb_offset, src_tp_for_af[i], self->m_bufs[i].get_arrmeta(), src_tp[i],
                                          src_arrmeta[i], kernreq, ectx);
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
      if (i < nsrc - 1) {
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<self_type>(root_ckb_offset);
      }
    }
  }

  return ckb_offset;
}
