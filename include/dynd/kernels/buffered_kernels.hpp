//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/buffer_storage.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  /**
   * Instantiate an arrfunc, adding buffers for any inputs where the types
   * don't match.
   */
  struct buffered_kernel
      : base_kernel<buffered_kernel, kernel_request_host, -1> {
    typedef buffered_kernel self_type;
    intptr_t m_nsrc;
    std::vector<intptr_t> m_src_buf_ck_offsets;
    std::vector<buffer_storage> m_bufs;

    buffered_kernel(intptr_t nsrc) : m_nsrc(nsrc) {}

    void single(char *dst, char *const *src)
    {
      std::vector<char *> buf_src(m_nsrc);
      for (intptr_t i = 0; i < m_nsrc; ++i) {
        if (!m_bufs[i].is_null()) {
          m_bufs[i].reset_arrmeta();
          ckernel_prefix *ck = get_child_ckernel(m_src_buf_ck_offsets[i]);
          expr_single_t ck_fn = ck->get_function<expr_single_t>();
          ck_fn(m_bufs[i].get_storage(), &src[i], ck);
          buf_src[i] = m_bufs[i].get_storage();
        } else {
          buf_src[i] = src[i];
        }
      }
      ckernel_prefix *child = get_child_ckernel();
      expr_single_t child_fn = child->get_function<expr_single_t>();
      child_fn(dst, &buf_src[0], child);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      std::vector<char *> buf_src(m_nsrc);
      std::vector<intptr_t> buf_stride(m_nsrc);
      ckernel_prefix *child = get_child_ckernel();
      expr_strided_t child_fn = child->get_function<expr_strided_t>();

      for (intptr_t i = 0; i < m_nsrc; ++i) {
        if (!m_bufs[i].is_null()) {
          buf_src[i] = m_bufs[i].get_storage();
          buf_stride[i] = m_bufs[i].get_stride();
        } else {
          buf_src[i] = src[i];
          buf_stride[i] = src_stride[i];
        }
      }

      while (count > 0) {
        size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
        for (intptr_t i = 0; i < m_nsrc; ++i) {
          if (!m_bufs[i].is_null()) {
            m_bufs[i].reset_arrmeta();
            ckernel_prefix *ck = get_child_ckernel(m_src_buf_ck_offsets[i]);
            expr_strided_t ck_fn = ck->get_function<expr_strided_t>();
            ck_fn(m_bufs[i].get_storage(), m_bufs[i].get_stride(), &src[i],
                  &src_stride[i], chunk_size, ck);
          }
        }
        child_fn(dst, dst_stride, &buf_src[0], &buf_stride[0], chunk_size,
                 child);
        for (intptr_t i = 0; i < m_nsrc; ++i) {
          if (!m_bufs[i].is_null()) {
            m_bufs[i].reset_arrmeta();
            ckernel_prefix *ck = get_child_ckernel(m_src_buf_ck_offsets[i]);
            expr_strided_t ck_fn = ck->get_function<expr_strided_t>();
            ck_fn(m_bufs[i].get_storage(), buf_stride[i], &src[i],
                  &src_stride[i], chunk_size, ck);
          } else {
            buf_src[i] += chunk_size * buf_stride[i];
          }
        }
        count -= chunk_size;
      }
    }
  };

  size_t make_buffered_ckernel(
      const arrfunc_type_data *af, const arrfunc_type *af_tp, void *ckb,
      intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
      intptr_t nsrc, const ndt::type *src_tp, const ndt::type *src_tp_for_af,
      const char *const *src_arrmeta, kernel_request_t kernreq,
      const eval::eval_context *ectx);

} // namespace dynd::nd
} // namespace dynd