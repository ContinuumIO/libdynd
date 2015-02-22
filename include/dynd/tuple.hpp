//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>
#include <dynd/types/tuple_type.hpp>

namespace dynd {
namespace nd {

  namespace detail {

    template <copy_request_t copyreq>
    struct fill_tuple_forward_value {
      template <size_t I, typename... T>
      void operator()(const ndt::type *tp, char *arrmeta,
                      const uintptr_t *arrmeta_offsets, char *data,
                      const uintptr_t *data_offsets, T &&... values) const
      {
        fill_forward_value(tp[I], arrmeta + arrmeta_offsets[I],
                           data + data_offsets[I],
                           get<I>(std::forward<T>(values)...));
      }
    };

  } // namespace dynd::nd::detail

  template <copy_request_t copyreq = copy_if_small, typename... T>
  array tuple(T &&... values)
  {
    typedef make_index_sequence<sizeof...(T)> I;

    ndt::type tp = ndt::make_tuple({ndt::forward_type_of(values)...}, false);

    array res = empty_shell(tp);
    tuple_type::fill_default_data_offsets(
        sizeof...(T), tp.extended<base_tuple_type>()->get_field_types_raw(),
        reinterpret_cast<uintptr_t *>(res.get_arrmeta()));

    index_proxy<I>::for_each(
        detail::fill_tuple_forward_value<copyreq>(),
        tp.extended<base_tuple_type>()->get_field_types_raw(),
        res.get_arrmeta(),
        res.get_type().extended<base_tuple_type>()->get_arrmeta_offsets_raw(),
        res.get_readwrite_originptr(),
        res.get_type().extended<base_tuple_type>()->get_data_offsets(
            res.get_arrmeta()),
        std::forward<T>(values)...);

    return res;
  }

} // namespace dynd::nd
} // namespace dynd