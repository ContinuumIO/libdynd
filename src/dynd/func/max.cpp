//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/max_kernel.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/func/max.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::max::make()
{
  typedef type_id_sequence<bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id, int128_type_id,
                           uint8_type_id, uint16_type_id, uint32_type_id, uint64_type_id, uint128_type_id,
                           float32_type_id, float64_type_id> comparable_type_ids;

  auto children = callable::make_all<max_kernel, comparable_type_ids>();

  return functional::reduction(functional::multidispatch(
      ndt::callable_type::make(ndt::scalar_kind_type::make(),
                               ndt::scalar_kind_type::make()),
      [children](const ndt::type & DYND_UNUSED(dst_tp),
                 intptr_t DYND_UNUSED(nsrc),
                 const ndt::type * src_tp) mutable->callable &
  {
        callable &child = children[src_tp[0].get_type_id()];
        if (child.is_null()) {
          throw runtime_error("no suitable child found for nd::sum");
        }

        return child;
      },
      data_size_max(children)));
}

DYND_API struct nd::max nd::max;
