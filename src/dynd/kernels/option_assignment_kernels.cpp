//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/option_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;

static intptr_t instantiate_option_as_value_assignment_kernel(
    char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, kernel_targets_t *DYND_UNUSED(targets), const eval::eval_context *ectx,
    intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
    const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  // In all cases not handled, we use the
  // regular S to T assignment kernel.
  //
  // Note that this does NOT catch the case where a value
  // which was ok with type S, but equals the NA
  // token in type T, is assigned. Checking this
  // properly across all the cases would add
  // fairly significant cost, and it seems maybe ok
  // to skip it.
  ndt::type val_dst_tp =
      dst_tp.get_type_id() == option_type_id ? dst_tp.extended<ndt::option_type>()->get_value_type() : dst_tp;
  ndt::type val_src_tp =
      src_tp[0].get_type_id() == option_type_id ? src_tp[0].extended<ndt::option_type>()->get_value_type() : src_tp[0];
  return ::make_assignment_kernel(ckb, ckb_offset, val_dst_tp, dst_arrmeta, val_src_tp, src_arrmeta[0], kernreq, ectx);
}

namespace {

struct option_callable_list {
  ndt::type af_tp[7];
  nd::base_callable af[7];

  option_callable_list()
  {
    int i = 0;
    af_tp[i] = ndt::type("(?string) -> ?S");
    af[i].instantiate = &nd::assignment_kernel<option_type_id, string_type_id>::instantiate;
    ++i;
    af_tp[i] = ndt::type("(?T) -> ?S");
    af[i].instantiate = &dynd::nd::assignment_kernel<option_type_id, option_type_id>::instantiate;
    ++i;
    af_tp[i] = ndt::type("(?T) -> S");
    af[i].instantiate = &dynd::nd::option_to_value_ck::instantiate;
    ++i;
    af_tp[i] = ndt::type("(string) -> ?S");
    af[i].instantiate = &nd::assignment_kernel<option_type_id, string_type_id>::instantiate;
    ++i;
    af_tp[i] = ndt::type("(float32) -> ?S");
    af[i].instantiate = &nd::assignment_kernel<option_type_id, float32_type_id>::instantiate;
    ++i;
    af_tp[i] = ndt::type("(float64) -> ?S");
    af[i].instantiate = &nd::assignment_kernel<option_type_id, float64_type_id>::instantiate;
    ++i;
    af_tp[i] = ndt::type("(T) -> S");
    af[i].instantiate = &instantiate_option_as_value_assignment_kernel;
  }

  inline intptr_t size() const { return sizeof(af) / sizeof(af[0]); }

  const nd::base_callable *get() const { return af; }
  const ndt::type *get_type() const { return af_tp; }
};
} // anonymous namespace

size_t kernels::make_option_assignment_kernel(void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                              const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
                                              kernel_request_t kernreq, const eval::eval_context *ectx)
{
  static option_callable_list afl;
  intptr_t size = afl.size();
  const nd::base_callable *af = afl.get();
  const ndt::callable_type *const *af_tp = reinterpret_cast<const ndt::callable_type *const *>(afl.get_type());
  map<std::string, ndt::type> typevars;
  for (intptr_t i = 0; i < size; ++i, ++af_tp, ++af) {
    typevars.clear();
    if ((*af_tp)->get_pos_type(0).match(src_tp, typevars) && (*af_tp)->get_return_type().match(dst_tp, typevars)) {
      return af->instantiate(NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, size, &src_tp, &src_arrmeta, kernreq,
                             NULL, ectx, 0, NULL, std::map<std::string, ndt::type>());
    }
  }

  stringstream ss;
  ss << "Could not instantiate option assignment kernel from " << src_tp << " to " << dst_tp;
  throw invalid_argument(ss.str());
}
