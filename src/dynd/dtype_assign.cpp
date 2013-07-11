//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/convert_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/diagnostics.hpp>

using namespace std;
using namespace dynd;

std::ostream& dynd::operator<<(ostream& o, assign_error_mode errmode)
{
    switch (errmode) {
        case assign_error_none:
            o << "none";
            break;
        case assign_error_overflow:
            o << "overflow";
            break;
        case assign_error_fractional:
            o << "fractional";
            break;
        case assign_error_inexact:
            o << "inexact";
            break;
        case assign_error_default:
            o << "default";
            break;
        default:
            o << "invalid error mode(" << (int)errmode << ")";
            break;
    }

    return o;
}

// Returns true if the destination dtype can represent *all* the values
// of the source dtype, false otherwise. This is used, for example,
// to skip any overflow checks when doing value assignments between differing
// types.
bool dynd::is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt)
{
    if (dst_dt.is_builtin() && src_dt.is_builtin()) {
        switch (src_dt.get_kind()) {
            case pattern_kind: // TODO: raise an error?
                return true;
            case bool_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        return true;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case int_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.get_data_size() >= src_dt.get_data_size();
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.get_data_size() > src_dt.get_data_size();
                    case complex_kind:
                        return dst_dt.get_data_size() > 2 * src_dt.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.get_data_size() > src_dt.get_data_size();
                    case uint_kind:
                        return dst_dt.get_data_size() >= src_dt.get_data_size();
                    case real_kind:
                        return dst_dt.get_data_size() > src_dt.get_data_size();
                    case complex_kind:
                        return dst_dt.get_data_size() > 2 * src_dt.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.get_data_size() >= src_dt.get_data_size();
                    case complex_kind:
                        return dst_dt.get_data_size() >= 2 * src_dt.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        return false;
                    case complex_kind:
                        return dst_dt.get_data_size() >= src_dt.get_data_size();
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_dt.get_kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        return false;
                    case bytes_kind:
                        return false;
                    default:
                        break;
                }
            case bytes_kind:
                return dst_dt.get_kind() == bytes_kind  &&
                        dst_dt.get_data_size() == src_dt.get_data_size();
            default:
                break;
        }

        throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
    }

    // Use the available base_type to check the casting
    if (!dst_dt.is_builtin()) {
        // Call with dst_dt (the first parameter) first
        return dst_dt.extended()->is_lossless_assignment(dst_dt, src_dt);
    } else {
        // Fall back to src_dt if the dst's extended is NULL
        return src_dt.extended()->is_lossless_assignment(dst_dt, src_dt);
    }
}

void dynd::dtype_copy(const ndt::type& dt,
                const char *dst_metadata, char *dst_data,
                const char *src_metadata, const char *src_data)
{
    size_t data_size = dt.get_data_size();
    if (dt.is_pod()) {
        memcpy(dst_data, src_data, data_size);
    } else {
        assignment_kernel k;
        make_assignment_kernel(&k, 0, dt, dst_metadata,
                        dt, src_metadata,
                        kernel_request_single,
                        assign_error_none, &eval::default_eval_context);
        k(dst_data, src_data);
    }
}

void dynd::dtype_assign(const ndt::type& dst_dt, const char *dst_metadata, char *dst_data,
                const ndt::type& src_dt, const char *src_metadata, const char *src_data,
                assign_error_mode errmode, const eval::eval_context *ectx)
{
    DYND_ASSERT_ALIGNED(dst, 0, dst_dt.get_data_alignment(), "dst dtype: " << dst_dt << ", src dtype: " << src_dt);
    DYND_ASSERT_ALIGNED(src, 0, src_dt.get_data_alignment(), "src dtype: " << src_dt << ", dst dtype: " << dst_dt);

    if (errmode == assign_error_default) {
        if (ectx != NULL) {
            errmode = ectx->default_assign_error_mode;
        } else if (dst_dt == src_dt) {
            errmode = assign_error_none;
        } else {
            stringstream ss;
            ss << "assignment from " << src_dt << " to " << dst_dt << " with default error mode requires an eval_context";
            throw runtime_error(ss.str());
        }
    }

    assignment_kernel k;
    make_assignment_kernel(&k, 0, dst_dt, dst_metadata,
                    src_dt, src_metadata,
                    kernel_request_single,
                    errmode, ectx);
    k(dst_data, src_data);
}
