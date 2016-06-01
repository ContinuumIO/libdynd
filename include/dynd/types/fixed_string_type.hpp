//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The fixed_string type represents a string with
// a particular encoding, stored in a fixed-size
// buffer.
//

#pragma once

#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API fixed_string_type : public base_string_type {
    intptr_t m_stringsize;
    const string_encoding_t m_encoding;
    const std::string m_encoding_repr;

  public:
    fixed_string_type(type_id_t id, intptr_t stringsize, string_encoding_t encoding = string_encoding_utf_8)
        : base_string_type(id, make_type<fixed_string_kind_type>(), 0, 1, type_flag_none, 0), m_stringsize(stringsize),
          m_encoding(encoding), m_encoding_repr(encoding_as_string(encoding)) {
      switch (encoding) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        this->m_data_size = m_stringsize;
        this->m_data_alignment = 1;
        break;
      case string_encoding_ucs_2:
      case string_encoding_utf_16:
        this->m_data_size = m_stringsize * 2;
        this->m_data_alignment = 2;
        break;
      case string_encoding_utf_32:
        this->m_data_size = m_stringsize * 4;
        this->m_data_alignment = 4;
        break;
      default:
        throw std::runtime_error("Unrecognized string encoding in dynd fixed_string type constructor");
      }
    }

    intptr_t get_size() const { return m_stringsize; }
    string_encoding_t get_encoding() const { return m_encoding; }

    void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    type get_canonical_type() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *DYND_UNUSED(arrmeta), bool DYND_UNUSED(blockref_alloc)) const {}
    void arrmeta_copy_construct(char *DYND_UNUSED(dst_arrmeta), const char *DYND_UNUSED(src_arrmeta),
                                const nd::memory_block &DYND_UNUSED(embedded_reference)) const {}
    void arrmeta_destruct(char *DYND_UNUSED(arrmeta)) const {}
    void arrmeta_debug_print(const char *DYND_UNUSED(arrmeta), std::ostream &DYND_UNUSED(o),
                             const std::string &DYND_UNUSED(indent)) const {}

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

  template <>
  struct id_of<fixed_string_type> : std::integral_constant<type_id_t, fixed_string_id> {};

} // namespace dynd::ndt
} // namespace dynd
