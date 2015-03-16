//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <memory>
#include <set>
#include <unordered_map>

#include <dynd/kernels/multidispatch.hpp>

using namespace std;
using namespace dynd;

namespace std {

template <>
struct hash<dynd::ndt::type> {
  size_t operator()(const ndt::type &tp) const { return tp.get_type_id(); }
};
template <>
struct hash<nd::string> {
  size_t operator()(const nd::string &) const { return 0; }
};
template <>
struct hash<std::vector<dynd::ndt::type>> {
  size_t operator()(const std::vector<dynd::ndt::type> &v) const
  {
    std::hash<ndt::type> hash;
    size_t value = 0;
    for (dynd::ndt::type tp : v) {
      value ^= hash(tp) + 0x9e3779b9 + (value << 6) + (value >> 2);
    }
    return value;
  }
};

} // namespace std

namespace dynd {
namespace nd {
  namespace functional {

    static const arrfunc_type_data *
    multidispatch_find(const arrfunc_type_data *self,
                       const std::map<string, ndt::type> &tp_vars)
    {
      const multidispatch_ck::data_type *data =
          self->get_data_as<multidispatch_ck::data_type>();
      std::shared_ptr<multidispatch_ck::map_type> map = data->map;
      std::shared_ptr<std::vector<string>> vars = data->vars;

      std::vector<ndt::type> tp_vals;
      for (auto pair : tp_vars) {
        if (std::find(vars->begin(), vars->end(), pair.first) != vars->end()) {
          tp_vals.push_back(pair.second);
        }
      }

      return (*map)[tp_vals].get();
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd

intptr_t nd::functional::multidispatch_ck::instantiate(
    const arrfunc_type_data *self, const arrfunc_type *self_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx, const array &kwds,
    const std::map<string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child = multidispatch_find(self, tp_vars);
  return child->instantiate(child, self_tp, ckb, ckb_offset, dst_tp,
                            dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq,
                            ectx, kwds, tp_vars);
}

void nd::functional::multidispatch_ck::resolve_option_values(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, intptr_t nsrc,
    const ndt::type *src_tp, nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child = multidispatch_find(self, tp_vars);
  if (child->resolve_option_values != NULL) {
    child->resolve_option_values(child, af_tp, nsrc, src_tp, kwds, tp_vars);
  }
}

int nd::functional::multidispatch_ck::resolve_dst_type(
    const arrfunc_type_data *self, const arrfunc_type *af_tp, intptr_t nsrc,
    const ndt::type *src_tp, int throw_on_error, ndt::type &out_dst_tp,
    const nd::array &kwds, const std::map<string, ndt::type> &tp_vars)
{
  const arrfunc_type_data *child = multidispatch_find(self, tp_vars);
  if (child->resolve_dst_type != NULL) {
    child->resolve_dst_type(child, af_tp, nsrc, src_tp, throw_on_error,
                            out_dst_tp, kwds, tp_vars);
  } else {
    out_dst_tp = af_tp->get_return_type();
  }

  out_dst_tp = ndt::substitute(out_dst_tp, tp_vars, true);

  return 1;
}