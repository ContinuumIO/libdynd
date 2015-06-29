//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/reduce.hpp>
#include <dynd/kernels/reduce.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/with.hpp>

using namespace std;
using namespace dynd;

namespace {

  /*
struct lifted_reduction_arrfunc_data {
  // Pointer to the child arrfunc
  nd::arrfunc child_elwise_reduction;
  nd::arrfunc child_dst_initialization;
  nd::array reduction_identity;
  // The types of the child ckernel and this one
  const ndt::type *child_data_types;
  ndt::type data_types[2];
  intptr_t reduction_ndim;
  bool associative, commutative, right_associative;
  shortvector<bool> reduction_dimflags;
};
*/

static void delete_lifted_reduction_arrfunc_data(arrfunc_type_data *self_af)
{
  auto self = *self_af->get_data_as<
      nd::functional::reduce_virtual_ck::resolution_data_type *>();
  delete self;
}

static intptr_t instantiate_lifted_reduction_arrfunc_data(
    const arrfunc_type_data *af_self, const ndt::arrfunc_type *af_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  auto data = *af_self->get_data_as<char *>();
  return nd::functional::reduce_virtual_ck::instantiate(
      af_self, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
      src_arrmeta, kernreq, ectx, kwds, tp_vars);
}

} // anonymous namespace

nd::arrfunc nd::functional::reduce(const nd::arrfunc &child,
                                   const ndt::type &lifted_arr_type,
                                   const nd::arrfunc &dst_initialization,
                                   bool keepdims, const nd::array &axis,
                                   bool associative, bool commutative,
                                   bool right_associative,
                                   const nd::array &reduction_identity)
{
  // Validate the input elwise_reduction arrfunc
  if (child.is_null()) {
    throw runtime_error("functional::reduce: 'child' may not be empty");
  }
  if (child.get()->resolve_dst_type == NULL) {
    throw std::runtime_error(
        "functional::reduce child has NULL resolve_dst_type");
  }
  if (child.get()->resolve_option_values == NULL) {
    throw std::runtime_error(
        "functional::reduce child has NULL resolve_option_values");
  }

  const ndt::arrfunc_type *elwise_reduction_tp = child.get_type();
  if (elwise_reduction_tp->get_npos() != 1 &&
      !(elwise_reduction_tp->get_npos() == 2 &&
        elwise_reduction_tp->get_pos_type(0) ==
            elwise_reduction_tp->get_pos_type(1) &&
        elwise_reduction_tp->get_pos_type(0) ==
            elwise_reduction_tp->get_return_type())) {
    stringstream ss;
    ss << "lift_reduction_arrfunc: 'elwise_reduction' must contain a"
          " unary operation ckernel or a binary expr ckernel with all "
          "equal types, its prototype is "
       << elwise_reduction_tp;
    throw invalid_argument(ss.str());
  }

  // TODO: better type here
  ndt::type lifted_dst_type;

  if (axis.is_null()) {
    // NULL axis means reduce all axes
    lifted_dst_type = child.get_return_type();
  } else if (ndt::type("Fixed * bool").match(axis.get_type())) {
    // TODO: inefficient construction of pattern in if condition above
    // An array of integer axes
    lifted_dst_type = child.get_return_type();
    nd::with_1d_stride<dynd::bool1>(axis, [&](intptr_t size, intptr_t stride,
                                              const dynd::bool1 *data) {
      for (intptr_t i = size - 1; i >= 0; --i) {
        if (data[i * stride]) {
          if (keepdims) {
            lifted_dst_type = ndt::make_fixed_dim(1, lifted_dst_type);
          }
        }
        else {
          ndt::type subtype = lifted_arr_type.get_type_at_dimension(NULL, i);
          switch (subtype.get_type_id()) {
          case fixed_dim_type_id:
            if (subtype.get_kind() == kind_kind) {
              lifted_dst_type = ndt::make_fixed_dim_kind(lifted_dst_type);
            }
            else {
              lifted_dst_type = ndt::make_fixed_dim(
                  subtype.extended<ndt::fixed_dim_type>()->get_fixed_dim_size(),
                  lifted_dst_type);
            }
            break;
          case var_dim_type_id:
            lifted_dst_type = ndt::make_var_dim(lifted_dst_type);
            break;
          default: {
            stringstream ss;
            ss << "lift_reduction_arrfunc: don't know how to process ";
            ss << "dimension of type " << subtype;
            throw type_error(ss.str());
          }
          }
        }
      }
    });
  } else {
    // Can't tell the # of dimensions in other cases
    lifted_dst_type = ndt::make_ellipsis_dim(child.get_return_type());
  }

  auto self = new nd::functional::reduce_virtual_ck::resolution_data_type;
  self->red_op = child;
  self->dst_init = dst_initialization;
  if (!reduction_identity.is_null()) {
    if (reduction_identity.is_immutable() &&
        reduction_identity.get_type() ==
            elwise_reduction_tp->get_return_type()) {
      self->red_ident = reduction_identity;
    }
    else {
      self->red_ident = nd::empty(elwise_reduction_tp->get_return_type());
      self->red_ident.vals() = reduction_identity;
      self->red_ident.flag_as_immutable();
    }
  }
  self->associative = associative;
  self->commutative = commutative;
  self->right_associative = right_associative;
  self->axis = axis;

  return nd::arrfunc(
      ndt::make_arrfunc(ndt::make_tuple(lifted_arr_type), lifted_dst_type),
      self, 0, &instantiate_lifted_reduction_arrfunc_data, NULL, NULL,
      &delete_lifted_reduction_arrfunc_data);
}
