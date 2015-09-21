//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/fill.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/call.hpp>
#include <dynd/kernels/zeros_kernel.hpp>
#include <dynd/kernels/ones_kernel.hpp>

using namespace std;
using namespace dynd;


DYND_API nd::callable nd::zeros::children[DYND_TYPE_ID_MAX + 1];

DYND_API nd::callable nd::zeros::make()
{
    typedef type_id_sequence<
        bool_type_id,
        int8_type_id,
        int16_type_id,
        int32_type_id,
        int64_type_id,
        int128_type_id,
        uint8_type_id,
        uint16_type_id,
        uint32_type_id,
        uint64_type_id,
        uint128_type_id,
        float16_type_id,
        float32_type_id,
        float64_type_id,
        float128_type_id,
        complex_float32_type_id,
        complex_float64_type_id
    > type_ids;
    for (auto &pair : callable::make_all<zeros_kernel, type_ids>()) {
        children[pair.first] = pair.second;
    }
    
    callable self = functional::call<zeros>(ndt::type("() -> Any"));
    
    for (auto tp_id : {fixed_dim_type_id, var_dim_type_id}) {
        children[tp_id] = functional::elwise(self);
    }
    
    return functional::multidispatch(ndt::type("() -> Any"),
             [](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                const ndt::type *DYND_UNUSED(src_tp)) -> callable & {
                 callable &child = children[dst_tp.get_type_id()];
                 if (child.is_null()) {
                     throw std::runtime_error("no child found");
                 }
                 return child;
             },
         0);
}

DYND_API nd::callable nd::ones::children[DYND_TYPE_ID_MAX + 1];

DYND_API nd::callable nd::ones::make()
{
    typedef type_id_sequence<
        bool_type_id,
        int8_type_id,
        int16_type_id,
        int32_type_id,
        int64_type_id,
        int128_type_id,
        uint8_type_id,
        uint16_type_id,
        uint32_type_id,
        uint64_type_id,
        uint128_type_id,
        float16_type_id,
        float32_type_id,
        float64_type_id,
        float128_type_id,
        complex_float32_type_id,
        complex_float64_type_id
    > type_ids;
    for (auto &pair : callable::make_all<ones_kernel, type_ids>()) {
        children[pair.first] = pair.second;
    }
    
    callable self = functional::call<ones>(ndt::type("() -> Any"));
    
    for (auto tp_id : {fixed_dim_type_id, var_dim_type_id}) {
        children[tp_id] = functional::elwise(self);
    }
    
    return functional::multidispatch(ndt::type("() -> Any"),
             [](const ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
                const ndt::type *DYND_UNUSED(src_tp)) -> callable & {
                 callable &child = children[dst_tp.get_type_id()];
                 if (child.is_null()) {
                     throw std::runtime_error("no child found");
                 }
                 return child;
             },
         0);
}