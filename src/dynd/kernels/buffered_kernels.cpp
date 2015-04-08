//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/buffered_kernels.hpp>

using namespace std;
using namespace dynd;

size_t nd::make_buffered_ckernel(
    const arrfunc_type_data *af, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t nsrc, const ndt::type *src_tp, const ndt::type *src_tp_for_af,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  typedef buffered_kernel self_type;
  intptr_t root_ckb_offset = ckb_offset;
  self_type *self = self_type::create(ckb, kernreq, ckb_offset, nsrc);
  // Prepare the type and buffer info the ckernel needs
  self->m_bufs.resize(nsrc);
  self->m_src_buf_ck_offsets.resize(nsrc);
  vector<const char *> buffered_arrmeta(nsrc);
  for (intptr_t i = 0; i < nsrc; ++i) {
    if (src_tp[i] == src_tp_for_af[i]) {
      buffered_arrmeta[i] = src_arrmeta[i];
    } else {
      self->m_bufs[i].allocate(src_tp_for_af[i]);
      buffered_arrmeta[i] = self->m_bufs[i].get_arrmeta();
    }
  }
  // Instantiate the arrfunc being buffered
  ckb_offset =
      af->instantiate(af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
                      nsrc, src_tp_for_af, &buffered_arrmeta[0], kernreq, ectx,
                      nd::array(), std::map<nd::string, ndt::type>());
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
      ->ensure_capacity(ckb_offset);
  self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
             ->get_at<self_type>(root_ckb_offset);
  // Instantiate assignments for all the buffered operands
  for (intptr_t i = 0; i < nsrc; ++i) {
    if (!self->m_bufs[i].is_null()) {
      self->m_src_buf_ck_offsets[i] = ckb_offset - root_ckb_offset;
      ckb_offset =
          make_assignment_kernel(NULL, NULL, ckb, ckb_offset, src_tp_for_af[i],
                                 self->m_bufs[i].get_arrmeta(), src_tp[i],
                                 src_arrmeta[i], kernreq, ectx, nd::array());
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->ensure_capacity(ckb_offset);
      if (i < nsrc - 1) {
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                   ->get_at<self_type>(root_ckb_offset);
      }
    }
  }

  return ckb_offset;
}
