//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/gfunc/call_gcallable.hpp>
#include <dynd/func/neighborhood.hpp>
#include <dynd/kernels/neighborhood.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::neighborhood(const callable &neighborhood_op, const callable &DYND_UNUSED(boundary_child))
{
  const ndt::callable_type *funcproto_tp = neighborhood_op.get_array_type().extended<ndt::callable_type>();

  intptr_t nh_ndim = funcproto_tp->get_pos_type(0).get_ndim();
  nd::array arg_tp = nd::empty(2, ndt::make_type());
  arg_tp(0).vals() = ndt::type("?" + std::to_string(nh_ndim) + " * int");
  arg_tp(1).vals() = ndt::type("?" + std::to_string(nh_ndim) + " * int");

  return callable::make<neighborhood_kernel<1>>(
      ndt::callable_type::make(funcproto_tp->get_pos_type(0).with_replaced_dtype(funcproto_tp->get_return_type()),
                               funcproto_tp->get_pos_tuple(), ndt::struct_type::make({"shape", "offset"}, arg_tp)),
      neighborhood_op, sizeof(neighborhood_kernel<1>::data_type));
}
