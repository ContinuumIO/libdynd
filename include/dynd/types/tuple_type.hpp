//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API tuple_type : public base_type {
  protected:
    /**
     * The number of values in m_field_types and m_arrmeta_offsets.
     */
    intptr_t m_field_count;

    /**
     * Immutable vector of field types.
     */
    std::vector<type> m_field_types;

    /**
     * Immutable vector of arrmeta offsets.
     */
    std::vector<uintptr_t> m_arrmeta_offsets;

    /**
     * If true, the tuple is variadic, which means it is symbolic, and matches
     * against the beginning of a concrete tuple.
     */
    bool m_variadic;

  public:
    tuple_type(type_id_t id, size_t size, const type *element_tp, bool variadic = false,
               uint32_t flags = type_flag_none)
        : base_type(id, make_type<scalar_kind_type>(), 0, 1,
                    flags | type_flag_indexable | (variadic ? type_flag_symbolic : 0), 0, 0, 0),
          m_field_count(size), m_field_types(size), m_arrmeta_offsets(size), m_variadic(variadic) {
      // Calculate the needed element alignment and arrmeta offsets
      size_t arrmeta_offset = get_field_count() * sizeof(size_t);

      this->m_data_alignment = 1;

      for (intptr_t i = 0; i < m_field_count; ++i) {
        m_field_types[i] = element_tp[i];
      }

      for (intptr_t i = 0; i != m_field_count; ++i) {
        const type &ft = get_field_type(i);
        size_t field_alignment = ft.get_data_alignment();
        // Accumulate the biggest field alignment as the type alignment
        if (field_alignment > this->m_data_alignment) {
          this->m_data_alignment = (uint8_t)field_alignment;
        }
        // Inherit any operand flags from the fields
        this->flags |= (ft.get_flags() & type_flags_operand_inherited);
        // Calculate the arrmeta offsets
        m_arrmeta_offsets[i] = arrmeta_offset;
        arrmeta_offset += ft.get_arrmeta_size();
      }

      this->m_metadata_size = arrmeta_offset;
    }

    tuple_type(type_id_t id, std::initializer_list<type> element_tp, bool variadic = false)
        : tuple_type(id, element_tp.size(), element_tp.begin(), variadic, type_flag_none) {}

    tuple_type(type_id_t id, bool variadic = false) : tuple_type(id, 0, nullptr, variadic, type_flag_none) {}

    intptr_t get_field_count() const { return m_field_count; }
    const ndt::type get_type() const { return ndt::type_for(m_field_types); }
    const std::vector<type> &get_field_types() const { return m_field_types; }
    const type *get_field_types_raw() const { return m_field_types.data(); }
    const std::vector<uintptr_t> &get_arrmeta_offsets() const { return m_arrmeta_offsets; }
    const uintptr_t *get_arrmeta_offsets_raw() const { return m_arrmeta_offsets.data(); }

    const type &get_field_type(intptr_t i) const { return m_field_types[i]; }
    uintptr_t get_arrmeta_offset(intptr_t i) const { return m_arrmeta_offsets[i]; }

    bool is_variadic() const { return m_variadic; }

    virtual void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;
    bool is_expression() const;
    bool is_unique_data_owner(const char *arrmeta) const;

    size_t get_default_data_size() const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;

    void print_type(std::ostream &o) const;

    void transform_child_types(type_transform_fn_t transform_fn, intptr_t arrmeta_offset, void *extra,
                               type &out_transformed_tp, bool &out_was_transformed) const;
    type get_canonical_type() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;

    void foreach_leading(const char *arrmeta, char *data, foreach_fn_t callback, void *callback_data) const;

    void arrmeta_debug_print(const char *arrmeta, std::ostream &o, const std::string &indent) const;

    virtual bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    /**
     * Fills in the array of default data offsets based on the data sizes
     * and alignments of the types.
     */
    static void fill_default_data_offsets(intptr_t nfields, const type *field_tps, uintptr_t *out_data_offsets) {
      if (nfields > 0) {
        out_data_offsets[0] = 0;
        size_t offs = 0;
        for (intptr_t i = 1; i < nfields; ++i) {
          offs += field_tps[i - 1].get_default_data_size();
          offs = inc_to_alignment(offs, field_tps[i].get_data_alignment());
          out_data_offsets[i] = offs;
        }
      }
    }
  };

  template <>
  struct id_of<tuple_type> : std::integral_constant<type_id_t, tuple_id> {};

} // namespace dynd::ndt
} // namespace dynd
