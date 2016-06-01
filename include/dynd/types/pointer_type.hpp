//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/**
 * The pointer type contains C/C++ raw pointers
 * pointing at data in other memory_blocks, using
 * blockrefs to manage the memory.
 *
 * This type operates in a "gather/scatter" fashion,
 * exposing itself as an expression type whose expression
 * copies the data to/from the pointer targets.
 */

#pragma once

#include <dynd/buffer.hpp>
#include <dynd/type.hpp>
#include <dynd/types/any_kind_type.hpp>

namespace dynd {

struct DYNDT_API pointer_type_arrmeta {
  /**
   * A reference to the memory block which contains the data.
   */
  nd::memory_block blockref;
  /* Each pointed-to destination is offset by this amount */
  intptr_t offset;
};

namespace ndt {

  class DYNDT_API pointer_type : public base_type {
    type m_target_tp;

  public:
    pointer_type(type_id_t id, const type &target_tp = make_type<any_kind_type>())
        : base_type(id, make_type<any_kind_type>(), sizeof(char *), alignof(char *),
                    target_tp.get_flags() | type_flag_zeroinit | type_flag_blockref,
                    target_tp.get_arrmeta_size() + sizeof(pointer_type_arrmeta), 0, 0),
          m_target_tp(target_tp) {}

    const type &get_value_type() const { return m_target_tp.value_type(); }
    const type &get_operand_type() const;

    const type &get_target_type() const { return m_target_tp; }

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    inline bool is_type_subarray(const type &subarray_tp) const {
      // Uniform dimensions can share one implementation
      return (!subarray_tp.is_builtin() && (*this) == (*subarray_tp.extended())) ||
             m_target_tp.is_type_subarray(subarray_tp);
    }

    bool is_unique_data_owner(const char *arrmeta) const;
    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;
    type at_single(intptr_t i0, const char **inout_arrmeta, const char **inout_data) const;

    type get_type_at_dimension(char **inout_arrmeta, intptr_t i, intptr_t total_ndim = 0) const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    axis_order_classification_t classify_axis_order(const char *arrmeta) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    type with_replaced_storage_type(const type &replacement_type) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

  template <>
  struct id_of<pointer_type> : std::integral_constant<type_id_t, pointer_id> {};

  template <typename T>
  struct traits<T *> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return make_type<pointer_type>(make_type<T>()); }
  };

} // namespace dynd::ndt
} // namespace dynd
