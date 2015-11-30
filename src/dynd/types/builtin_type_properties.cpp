//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/gfunc/make_gcallable.hpp>

using namespace std;
using namespace dynd;

namespace dynd {
namespace nd {

  struct complex_real_kernel : nd::base_kernel<complex_real_kernel> {
    static const size_t data_size = 0;

    array self;

    complex_real_kernel(const array &self) : self(self) {}

    void call(array *dst, array *const *DYND_UNUSED(src)) { *dst = helper(self); }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0]).get_type();
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      complex_real_kernel::make(ckb, kernreq, ckb_offset, kwds[0]);
      return ckb_offset;
    }

    static nd::array helper(const nd::array &n)
    {
      return n.replace_dtype(ndt::property_type::make(n.get_dtype(), "real"));
    }
  };

  struct complex_imag_kernel : nd::base_kernel<complex_imag_kernel> {
    static const size_t data_size = 0;

    array self;

    complex_imag_kernel(const array &self) : self(self) {}

    void call(array *dst, array *const *DYND_UNUSED(src)) { *dst = helper(self); }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0]).get_type();
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      complex_imag_kernel::make(ckb, kernreq, ckb_offset, kwds[0]);
      return ckb_offset;
    }

    static nd::array helper(const nd::array &n)
    {
      return n.replace_dtype(ndt::property_type::make(n.get_dtype(), "imag"));
    }
  };

  struct complex_conj_kernel : nd::base_kernel<complex_conj_kernel> {
    static const size_t data_size = 0;

    array self;

    complex_conj_kernel(const array &self) : self(self) {}

    void call(array *dst, array *const *DYND_UNUSED(src)) { *dst = helper(self); }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                 intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                 intptr_t DYND_UNUSED(nkwd), const array *kwds,
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = helper(kwds[0]).get_type();
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      complex_conj_kernel::make(ckb, kernreq, ckb_offset, kwds[0]);
      return ckb_offset;
    }

    static nd::array helper(const nd::array &n)
    {
      return n.replace_dtype(ndt::property_type::make(n.get_dtype(), "conj"));
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::complex_real_kernel> {
    static type make() { return type("(self: Any) -> Any"); }
  };

  template <>
  struct type::equivalent<nd::complex_imag_kernel> {
    static type make() { return type("(self: Any) -> Any"); }
  };

  template <>
  struct type::equivalent<nd::complex_conj_kernel> {
    static type make() { return type("(self: Any) -> Any"); }
  };

} // namespace dynd::ndt
} // namespace dynd

static nd::array property_complex_real(const nd::array &n)
{
  nd::callable f = nd::callable::make<nd::complex_real_kernel>();
  return f(kwds("self", n));
}

static nd::array property_complex_imag(const nd::array &n)
{
  nd::callable f = nd::callable::make<nd::complex_imag_kernel>();
  return f(kwds("self", n));
}

static nd::array property_complex_conj(const nd::array &n)
{
  nd::callable f = nd::callable::make<nd::complex_conj_kernel>();
  return f(kwds("self", n));
}

static size_t complex_array_properties_size() { return 3; }

static const pair<std::string, gfunc::callable> *complex_array_properties()
{
  static const pair<std::string, gfunc::callable> complex_array_properties[3] = {
      pair<std::string, gfunc::callable>("real", gfunc::make_callable(&property_complex_real, "self")),
      pair<std::string, gfunc::callable>("imag", gfunc::make_callable(&property_complex_imag, "self")),
      pair<std::string, gfunc::callable>("conj", gfunc::make_callable(&property_complex_conj, "self"))};

  return complex_array_properties;
}

void dynd::get_builtin_type_dynamic_array_properties(type_id_t builtin_type_id,
                                                     const std::pair<std::string, gfunc::callable> **out_properties,
                                                     size_t *out_count)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
  case complex_float64_type_id:
    *out_properties = complex_array_properties();
    *out_count = complex_array_properties_size();
    break;
  default:
    *out_properties = NULL;
    *out_count = 0;
    break;
  }
}

size_t dynd::get_builtin_type_elwise_property_index(type_id_t builtin_type_id, const std::string &property_name)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
  case complex_float64_type_id:
    if (property_name == "real") {
      return 0;
    }
    else if (property_name == "imag") {
      return 1;
    }
    else if (property_name == "conj") {
      return 2;
    }
  default:
    break;
  }

  std::stringstream ss;
  ss << "the dynd type " << ndt::type(builtin_type_id);
  ss << " doesn't have a property \"" << property_name << "\"";
  throw std::runtime_error(ss.str());
}

ndt::type dynd::get_builtin_type_elwise_property_type(type_id_t builtin_type_id, size_t elwise_property_index,
                                                      bool &out_readable, bool &out_writable)
{
  switch (builtin_type_id) {
  case complex_float32_type_id:
    switch (elwise_property_index) {
    case 0: // real
    case 1: // imag
      out_readable = true;
      out_writable = false;
      return ndt::type(float32_type_id);
    case 2: // conj
      out_readable = true;
      out_writable = true;
      return ndt::type(complex_float32_type_id);
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (elwise_property_index) {
    case 0: // real
    case 1: // imag
      out_readable = true;
      out_writable = false;
      return ndt::type(float64_type_id);
    case 2: // conj
      out_readable = true;
      out_writable = true;
      return ndt::type(complex_float64_type_id);
    default:
      break;
    }
    break;
  default:
    break;
  }
  out_readable = false;
  out_writable = false;
  return ndt::type();
}

static void get_property_kernel_complex_float32_real(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
{
  *reinterpret_cast<uint32_t *>(dst) = (*reinterpret_cast<uint32_t *const *>(src))[0];
}

static void get_property_kernel_complex_float32_imag(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
{
  *reinterpret_cast<uint32_t *>(dst) = (*reinterpret_cast<uint32_t *const *>(src))[1];
}

static void get_property_kernel_complex_float64_real(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
{
  *reinterpret_cast<uint64_t *>(dst) = (*reinterpret_cast<uint64_t *const *>(src))[0];
}

static void get_property_kernel_complex_float64_imag(ckernel_prefix *DYND_UNUSED(self), char *dst, char *const *src)
{
  *reinterpret_cast<uint64_t *>(dst) = (*reinterpret_cast<uint64_t *const *>(src))[1];
}

static void get_or_set_property_kernel_complex_float32_conj(ckernel_prefix *DYND_UNUSED(self), char *dst,
                                                            char *const *src)
{
  dynd::complex<float> value = **reinterpret_cast<dynd::complex<float> *const *>(src);
  *reinterpret_cast<dynd::complex<float> *>(dst) = dynd::complex<float>(value.real(), -value.imag());
}

static void get_or_set_property_kernel_complex_float64_conj(ckernel_prefix *DYND_UNUSED(self), char *dst,
                                                            char *const *src)
{
  dynd::complex<double> value = **reinterpret_cast<dynd::complex<double> *const *>(src);
  *reinterpret_cast<dynd::complex<double> *>(dst) = dynd::complex<double>(value.real(), -value.imag());
}

size_t dynd::make_builtin_type_elwise_property_getter_kernel(void *ckb, intptr_t ckb_offset, type_id_t builtin_type_id,
                                                             const char *DYND_UNUSED(dst_arrmeta),
                                                             const char *DYND_UNUSED(src_arrmeta),
                                                             size_t src_elwise_property_index, kernel_request_t kernreq,
                                                             const eval::eval_context *DYND_UNUSED(ectx))
{
  ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
  ckernel_prefix *e =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<ckernel_prefix>(ckb_offset);
  switch (builtin_type_id) {
  case complex_float32_type_id:
    switch (src_elwise_property_index) {
    case 0:
      e->function = reinterpret_cast<void *>(&get_property_kernel_complex_float32_real);
      return ckb_offset;
    case 1:
      e->function = reinterpret_cast<void *>(&get_property_kernel_complex_float32_imag);
      return ckb_offset;
    case 2:
      e->function = reinterpret_cast<void *>(&get_or_set_property_kernel_complex_float32_conj);
      return ckb_offset;
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (src_elwise_property_index) {
    case 0:
      e->function = reinterpret_cast<void *>(&get_property_kernel_complex_float64_real);
      return ckb_offset;
    case 1:
      e->function = reinterpret_cast<void *>(&get_property_kernel_complex_float64_imag);
      return ckb_offset;
    case 2:
      e->function = reinterpret_cast<void *>(&get_or_set_property_kernel_complex_float64_conj);
      return ckb_offset;
    default:
      break;
    }
    break;
  default:
    break;
  }
  stringstream ss;
  ss << "dynd type " << ndt::type(builtin_type_id) << " given an invalid property index " << src_elwise_property_index;
  throw runtime_error(ss.str());
}

size_t dynd::make_builtin_type_elwise_property_setter_kernel(void *ckb, intptr_t ckb_offset, type_id_t builtin_type_id,
                                                             const char *DYND_UNUSED(dst_arrmeta),
                                                             size_t dst_elwise_property_index,
                                                             const char *DYND_UNUSED(src_arrmeta),
                                                             kernel_request_t kernreq,
                                                             const eval::eval_context *DYND_UNUSED(ectx))
{
  ckb_offset = make_kernreq_to_single_kernel_adapter(ckb, ckb_offset, 1, kernreq);
  ckernel_prefix *e =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck<ckernel_prefix>(ckb_offset);
  switch (builtin_type_id) {
  case complex_float32_type_id:
    switch (dst_elwise_property_index) {
    case 2:
      e->function = reinterpret_cast<void *>(&get_or_set_property_kernel_complex_float32_conj);
      return ckb_offset;
    default:
      break;
    }
    break;
  case complex_float64_type_id:
    switch (dst_elwise_property_index) {
    case 2:
      e->function = reinterpret_cast<void *>(&get_or_set_property_kernel_complex_float64_conj);
      return ckb_offset;
    default:
      break;
    }
    break;
  default:
    break;
  }
  stringstream ss;
  ss << "dynd type " << ndt::type(builtin_type_id) << " given an invalid property index " << dst_elwise_property_index;
  throw runtime_error(ss.str());
}
