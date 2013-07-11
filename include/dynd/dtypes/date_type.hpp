//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_TYPE_HPP_
#define _DYND__DATE_TYPE_HPP_

#include <dynd/type.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/view_type.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

#define DYND_DATE_NA (std::numeric_limits<int32_t>::min())


class date_type : public base_type {
public:
    date_type();

    virtual ~date_type();

    // A static instance of a struct dtype used by default for a date
    static const ndt::type default_struct_type;

    void set_ymd(const char *metadata, char *data, assign_error_mode errmode,
                    int32_t year, int32_t month, int32_t day) const;
    void set_utf8_string(const char *metadata, char *data, assign_error_mode errmode, const std::string& utf8_str) const;

    void get_ymd(const char *metadata, const char *data,
                    int32_t &out_year, int32_t &out_month, int32_t &out_day) const;

    void print_data(std::ostream& o, const char *metadata, const char *data) const;

    void print_dtype(std::ostream& o) const;

    bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt) const;

    bool operator==(const base_type& rhs) const;

    void metadata_default_construct(char *DYND_UNUSED(metadata), size_t DYND_UNUSED(ndim), const intptr_t* DYND_UNUSED(shape)) const {
    }
    void metadata_copy_construct(char *DYND_UNUSED(dst_metadata), const char *DYND_UNUSED(src_metadata), memory_block_data *DYND_UNUSED(embedded_reference)) const {
    }
    void metadata_destruct(char *DYND_UNUSED(metadata)) const {
    }
    void metadata_debug_print(const char *DYND_UNUSED(metadata), std::ostream& DYND_UNUSED(o), const std::string& DYND_UNUSED(indent)) const {
    }

    size_t make_assignment_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const ndt::type& dst_dt, const char *dst_metadata,
                    const ndt::type& src_dt, const char *src_metadata,
                    kernel_request_t kernreq, assign_error_mode errmode,
                    const eval::eval_context *ectx) const;

    size_t make_comparison_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const ndt::type& src0_dt, const char *src0_metadata,
                    const ndt::type& src1_dt, const char *src1_metadata,
                    comparison_type_t comptype,
                    const eval::eval_context *ectx) const;

    void get_dynamic_dtype_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_dtype_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;
    void get_dynamic_array_properties(const std::pair<std::string, gfunc::callable> **out_properties, size_t *out_count) const;
    void get_dynamic_array_functions(const std::pair<std::string, gfunc::callable> **out_functions, size_t *out_count) const;

    size_t get_elwise_property_index(const std::string& property_name) const;
    ndt::type get_elwise_property_type(size_t elwise_property_index,
                    bool& out_readable, bool& out_writable) const;
    size_t make_elwise_property_getter_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata,
                    const char *src_metadata, size_t src_elwise_property_index,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
    size_t make_elwise_property_setter_kernel(
                    hierarchical_kernel *out, size_t offset_out,
                    const char *dst_metadata, size_t dst_elwise_property_index,
                    const char *src_metadata,
                    kernel_request_t kernreq, const eval::eval_context *ectx) const;
};

inline ndt::type make_date_type() {
    return ndt::type(new date_type(), false);
}

} // namespace dynd

#endif // _DYND__DATE_TYPE_HPP_
