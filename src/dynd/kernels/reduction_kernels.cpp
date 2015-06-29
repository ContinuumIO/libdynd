//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/array.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>
#include <dynd/func/reduce.hpp>
#include <dynd/kernels/base_kernel.hpp>

using namespace std;
using namespace dynd;

namespace {
template <class T, class Accum>
struct sum_reduction
    : nd::base_kernel<sum_reduction<T, Accum>, kernel_request_host, 1> {
  void single(char *dst, char *const *src)
  {
    *reinterpret_cast<T *>(dst) =
        *reinterpret_cast<T *>(dst) + **reinterpret_cast<T *const *>(src);
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src,
               const intptr_t *src_stride, size_t count)
  {
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      Accum s = 0;
      for (size_t i = 0; i < count; ++i) {
        s = s + *reinterpret_cast<T *>(src0);
        src0 += src0_stride;
      }
      *reinterpret_cast<T *>(dst) =
          static_cast<T>(*reinterpret_cast<T *>(dst) + s);
    } else {
      for (size_t i = 0; i < count; ++i) {
        *reinterpret_cast<T *>(dst) =
            *reinterpret_cast<T *>(dst) + *reinterpret_cast<T *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }
};
} // anonymous namespace

intptr_t kernels::make_builtin_sum_reduction_ckernel(void *ckb,
                                                     intptr_t ckb_offset,
                                                     type_id_t tid,
                                                     kernel_request_t kernreq)
{
  switch (tid) {
  case int32_type_id:
    sum_reduction<int32_t, int32_t>::make(ckb, kernreq, ckb_offset);
    break;
  case int64_type_id:
    sum_reduction<int64_t, int64_t>::make(ckb, kernreq, ckb_offset);
    break;
  case float32_type_id:
    sum_reduction<float, double>::make(ckb, kernreq, ckb_offset);
    break;
  case float64_type_id:
    sum_reduction<double, double>::make(ckb, kernreq, ckb_offset);
    break;
  case complex_float32_type_id:
    sum_reduction<complex<float>, complex<float>>::make(ckb, kernreq,
                                                        ckb_offset);
    break;
  case complex_float64_type_id:
    sum_reduction<complex<double>, complex<double>>::make(ckb, kernreq,
                                                          ckb_offset);
    break;
  default: {
    stringstream ss;
    ss << "make_builtin_sum_reduction_ckernel: data type ";
    ss << ndt::type(tid) << " is not supported";
    throw type_error(ss.str());
  }
  }

  return ckb_offset;
}

static intptr_t instantiate_builtin_sum_reduction_arrfunc(
    const arrfunc_type_data *DYND_UNUSED(self_data_ptr),
    const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data), void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
    const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
    kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
    const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  if (dst_tp != src_tp[0]) {
    stringstream ss;
    ss << "dynd sum reduction: the source type, " << src_tp[0]
       << ", does not match the destination type, " << dst_tp;
    throw type_error(ss.str());
  }
  return kernels::make_builtin_sum_reduction_ckernel(
      ckb, ckb_offset, dst_tp.get_type_id(), kernreq);
}

nd::arrfunc kernels::make_builtin_sum_reduction_arrfunc(type_id_t tid)
{
  if (tid < 0 || tid >= builtin_type_id_count) {
    stringstream ss;
    ss << "make_builtin_sum_reduction_ckernel: data type ";
    ss << ndt::type(tid) << " is not supported";
    throw type_error(ss.str());
  }
  nd::array af = nd::empty(
      ndt::make_arrfunc(ndt::make_tuple(ndt::type(tid)), ndt::type(tid)));
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  *out_af->get_data_as<type_id_t>() = tid;
  out_af->instantiate = &instantiate_builtin_sum_reduction_arrfunc;
  out_af->free = NULL;
  af.flag_as_immutable();
  return af;
}

nd::arrfunc kernels::make_builtin_sum1d_arrfunc(type_id_t tid)
{
  nd::arrfunc sum_ew = kernels::make_builtin_sum_reduction_arrfunc(tid);
  return nd::functional::reduce(
      sum_ew, ndt::make_fixed_dim_kind(ndt::type(tid)), nd::array(), false,
      {true}, true, true, false, 0);
}

namespace {
struct double_mean1d_ck
    : nd::base_kernel<double_mean1d_ck, kernel_request_host, 1> {
  intptr_t m_minp;
  intptr_t m_src_dim_size, m_src_stride;

  void single(char *dst, char *const *src)
  {
    intptr_t minp = m_minp, countp = 0;
    intptr_t src_dim_size = m_src_dim_size, src_stride = m_src_stride;
    double result = 0;
    char *src_copy = src[0];
    for (intptr_t i = 0; i < src_dim_size; ++i) {
      double v = *reinterpret_cast<double *>(src_copy);
      if (!dynd::isnan(v)) {
        result += v;
        ++countp;
      }
      src_copy += src_stride;
    }
    if (countp >= minp) {
      *reinterpret_cast<double *>(dst) = result / countp;
    } else {
      *reinterpret_cast<double *>(dst) = numeric_limits<double>::quiet_NaN();
    }
  }
};

struct mean1d_arrfunc_data {
  intptr_t minp;

  static void free(arrfunc_type_data *self_af)
  {
    delete *self_af->get_data_as<mean1d_arrfunc_data *>();
  }

  static intptr_t
  instantiate(const arrfunc_type_data *af_self,
              const ndt::arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
              void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
              const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
              const ndt::type *src_tp, const char *const *src_arrmeta,
              kernel_request_t kernreq,
              const eval::eval_context *DYND_UNUSED(ectx),
              const nd::array &DYND_UNUSED(kwds),
              const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
  {
    typedef double_mean1d_ck self_type;
    mean1d_arrfunc_data *data = *af_self->get_data_as<mean1d_arrfunc_data *>();
    self_type *self = self_type::make(ckb, kernreq, ckb_offset);
    intptr_t src_dim_size, src_stride;
    ndt::type src_el_tp;
    const char *src_el_arrmeta;
    if (!src_tp[0].get_as_strided(src_arrmeta[0], &src_dim_size, &src_stride,
                                  &src_el_tp, &src_el_arrmeta)) {
      stringstream ss;
      ss << "mean1d: could not process type " << src_tp[0];
      ss << " as a strided dimension";
      throw type_error(ss.str());
    }
    if (src_el_tp.get_type_id() != float64_type_id ||
        dst_tp.get_type_id() != float64_type_id) {
      stringstream ss;
      ss << "mean1d: input element type and output type must be "
            "float64, got " << src_el_tp << " and " << dst_tp;
      throw invalid_argument(ss.str());
    }
    self->m_minp = data->minp;
    if (self->m_minp <= 0) {
      if (self->m_minp <= -src_dim_size) {
        throw invalid_argument(
            "minp parameter is too large of a negative number");
      }
      self->m_minp += src_dim_size;
    }
    self->m_src_dim_size = src_dim_size;
    self->m_src_stride = src_stride;
    return ckb_offset;
  }
};
} // anonymous namespace

nd::arrfunc kernels::make_builtin_mean1d_arrfunc(type_id_t tid, intptr_t minp)
{
  if (tid != float64_type_id) {
    stringstream ss;
    ss << "make_builtin_mean1d_arrfunc: data type ";
    ss << ndt::type(tid) << " is not supported";
    throw type_error(ss.str());
  }
  nd::array mean1d = nd::empty(ndt::make_arrfunc(
      ndt::make_tuple(ndt::make_fixed_dim_kind(ndt::make_type<double>())),
      ndt::make_type<double>()));
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(mean1d.get_readwrite_originptr());
  mean1d_arrfunc_data *data = new mean1d_arrfunc_data;
  data->minp = minp;
  *out_af->get_data_as<mean1d_arrfunc_data *>() = data;
  out_af->instantiate = &mean1d_arrfunc_data::instantiate;
  out_af->free = &mean1d_arrfunc_data::free;
  mean1d.flag_as_immutable();
  return mean1d;
}
