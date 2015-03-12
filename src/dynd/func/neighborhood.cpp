//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/func/neighborhood.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/neighborhood.hpp>

using namespace std;
using namespace dynd;

static void free_neighborhood(arrfunc_type_data *self_af)
{
  neighborhood *nh = *self_af->get_data_as<neighborhood *>();
  free(nh->start_stop);
  delete nh;
}

static ndt::type make_neighborhood_type(const ndt::type &child_tp)
{
  intptr_t ndim = child_tp.extended<arrfunc_type>()->get_pos_type(0).get_ndim();

  ndt::type ret_tp =
      child_tp.extended<arrfunc_type>()->get_pos_type(0).with_replaced_dtype(
          child_tp.extended<arrfunc_type>()->get_return_type());

  ndt::type kwd_tp = ndt::make_struct(
      nd::array({"shape", "offset", "mask"}),
      {ndt::make_option(ndt::make_fixed_dim(ndim, ndt::make_type<intptr_t>())),
       ndt::make_option(ndt::make_fixed_dim(ndim, ndt::make_type<intptr_t>())),
       ndt::make_option(ndt::make_fixed_dimsym(ndt::make_type<bool>(), ndim))});

  return ndt::make_arrfunc(child_tp.extended<arrfunc_type>()->get_pos_tuple(),
                           kwd_tp, ret_tp);
}

nd::arrfunc nd::functional::neighborhood(const nd::arrfunc &child)
{
  const arrfunc_type *funcproto_tp = child.get_type();
  intptr_t ndim = funcproto_tp->get_pos_type(0).get_ndim();

  ndt::type self_tp = make_neighborhood_type(child.get_array_type());

  std::ostringstream oss;
  oss << "Fixed**" << ndim;
  ndt::type nhop_pattern("(" + oss.str() + " * NH) -> OUT");
  ndt::type result_pattern("(" + oss.str() + " * NH) -> " + oss.str() +
                           " * OUT");

  map<nd::string, ndt::type> typevars;
  if (!nhop_pattern.match(child.get_array_type(), typevars)) {
    stringstream ss;
    ss << "provided neighborhood op proto " << child.get_array_type()
       << " does not match pattern " << nhop_pattern;
    throw invalid_argument(ss.str());
  }

  nd::array af = nd::empty(self_tp);
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  struct neighborhood **nh = out_af->get_data_as<struct neighborhood *>();
  *nh = new struct neighborhood;
  (*nh)->op = child;
  (*nh)->start_stop = (start_stop_t *)malloc(ndim * sizeof(start_stop_t));
  out_af->instantiate = &neighborhood_ck<1>::instantiate;
  out_af->resolve_option_values = NULL;
  out_af->resolve_dst_type = &neighborhood_ck<1>::resolve_dst_type;
  out_af->free = &free_neighborhood;
  af.flag_as_immutable();
  return af;
}
