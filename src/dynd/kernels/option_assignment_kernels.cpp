//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/option_assignment_kernels.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace dynd;

namespace {

/**
 * A ckernel which assigns option[S] to option[T].
 */
struct option_to_option_ck
    : nd::base_kernel<option_to_option_ck, kernel_request_host, 1> {
  // The default child is the src is_avail ckernel
  // This child is the dst assign_na ckernel
  size_t m_dst_assign_na_offset;
  size_t m_value_assign_offset;

  void single(char *dst, char *const *src)
  {
    // Check whether the value is available
    // TODO: Would be nice to do this as a predicate
    //       instead of having to go through a dst pointer
    ckernel_prefix *src_is_avail = get_child_ckernel();
    expr_single_t src_is_avail_fn = src_is_avail->get_function<expr_single_t>();
    bool1 avail = bool1(false);
    src_is_avail_fn(reinterpret_cast<char *>(&avail), src, src_is_avail);
    if (avail) {
      // It's available, copy using value assignment
      ckernel_prefix *value_assign = get_child_ckernel(m_value_assign_offset);
      expr_single_t value_assign_fn =
          value_assign->get_function<expr_single_t>();
      value_assign_fn(dst, src, value_assign);
    } else {
      // It's not available, assign an NA
      ckernel_prefix *dst_assign_na = get_child_ckernel(m_dst_assign_na_offset);
      expr_single_t dst_assign_na_fn =
          dst_assign_na->get_function<expr_single_t>();
      dst_assign_na_fn(dst, NULL, dst_assign_na);
    }
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src,
               const intptr_t *src_stride, size_t count)
  {
    // Three child ckernels
    ckernel_prefix *src_is_avail = get_child_ckernel();
    expr_strided_t src_is_avail_fn =
        src_is_avail->get_function<expr_strided_t>();
    ckernel_prefix *value_assign = get_child_ckernel(m_value_assign_offset);
    expr_strided_t value_assign_fn =
        value_assign->get_function<expr_strided_t>();
    ckernel_prefix *dst_assign_na = get_child_ckernel(m_dst_assign_na_offset);
    expr_strided_t dst_assign_na_fn =
        dst_assign_na->get_function<expr_strided_t>();
    // Process in chunks using the dynd default buffer size
    bool1 avail[DYND_BUFFER_CHUNK_SIZE];
    while (count > 0) {
      size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
      count -= chunk_size;
      src_is_avail_fn(reinterpret_cast<char *>(avail), 1, src, src_stride,
                      chunk_size, src_is_avail);
      void *avail_ptr = avail;
      char *src_copy = src[0];
      do {
        // Process a run of available values
        void *next_avail_ptr = memchr(avail_ptr, 0, chunk_size);
        if (!next_avail_ptr) {
          value_assign_fn(dst, dst_stride, &src_copy, src_stride, chunk_size,
                          value_assign);
          dst += chunk_size * dst_stride;
          src += chunk_size * src_stride[0];
          break;
        } else if (next_avail_ptr > avail_ptr) {
          size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
          value_assign_fn(dst, dst_stride, &src_copy, src_stride, segment_size,
                          value_assign);
          dst += segment_size * dst_stride;
          src_copy += segment_size * src_stride[0];
          chunk_size -= segment_size;
          avail_ptr = next_avail_ptr;
        }
        // Process a run of not available values
        next_avail_ptr = memchr(avail_ptr, 1, chunk_size);
        if (!next_avail_ptr) {
          dst_assign_na_fn(dst, dst_stride, NULL, NULL, chunk_size,
                           dst_assign_na);
          dst += chunk_size * dst_stride;
          src_copy += chunk_size * src_stride[0];
          break;
        } else if (next_avail_ptr > avail_ptr) {
          size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
          dst_assign_na_fn(dst, dst_stride, NULL, NULL, segment_size,
                           dst_assign_na);
          dst += segment_size * dst_stride;
          src_copy += segment_size * src_stride[0];
          chunk_size -= segment_size;
          avail_ptr = next_avail_ptr;
        }
      } while (chunk_size > 0);
    }
  }

  void destruct_children()
  {
    // src_is_avail
    get_child_ckernel()->destroy();
    // dst_assign_na
    destroy_child_ckernel(m_dst_assign_na_offset);
    // value_assign
    destroy_child_ckernel(m_value_assign_offset);
  }
};

/**
 * A ckernel which assigns option[S] to T.
 */
struct option_to_value_ck
    : nd::base_kernel<option_to_value_ck, kernel_request_host, 1> {
  // The default child is the src_is_avail ckernel
  size_t m_value_assign_offset;

  void single(char *dst, char *const *src)
  {
    ckernel_prefix *src_is_avail = get_child_ckernel();
    expr_single_t src_is_avail_fn = src_is_avail->get_function<expr_single_t>();
    ckernel_prefix *value_assign = get_child_ckernel(m_value_assign_offset);
    expr_single_t value_assign_fn = value_assign->get_function<expr_single_t>();
    // Make sure it's not an NA
    bool1 avail = bool1(false);
    src_is_avail_fn(reinterpret_cast<char *>(&avail), src, src_is_avail);
    if (!avail) {
      throw overflow_error("cannot assign an NA value to a non-option type");
    }
    // Copy using value assignment
    value_assign_fn(dst, src, value_assign);
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src,
               const intptr_t *src_stride, size_t count)
  {
    // Two child ckernels
    ckernel_prefix *src_is_avail = get_child_ckernel();
    expr_strided_t src_is_avail_fn =
        src_is_avail->get_function<expr_strided_t>();
    ckernel_prefix *value_assign = get_child_ckernel(m_value_assign_offset);
    expr_strided_t value_assign_fn =
        value_assign->get_function<expr_strided_t>();
    // Process in chunks using the dynd default buffer size
    bool1 avail[DYND_BUFFER_CHUNK_SIZE];
    char *src_copy = src[0];
    while (count > 0) {
      size_t chunk_size = min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
      src_is_avail_fn(reinterpret_cast<char *>(avail), 1, &src_copy, src_stride,
                      chunk_size, src_is_avail);
      if (memchr(avail, 0, chunk_size) != NULL) {
        throw overflow_error("cannot assign an NA value to a non-option type");
      }
      value_assign_fn(dst, dst_stride, &src_copy, src_stride, chunk_size,
                      value_assign);
      dst += chunk_size * dst_stride;
      src_copy += chunk_size * src_stride[0];
      count -= chunk_size;
    }
  }

  void destruct_children()
  {
    // src_is_avail
    get_child_ckernel()->destroy();
    // value_assign
    destroy_child_ckernel(m_value_assign_offset);
  }
};

} // anonymous namespace

static intptr_t instantiate_option_to_option_assignment_kernel(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  intptr_t root_ckb_offset = ckb_offset;
  typedef option_to_option_ck self_type;
  if (dst_tp.get_type_id() != option_type_id ||
      src_tp[0].get_type_id() != option_type_id) {
    stringstream ss;
    ss << "option to option kernel needs option types, got " << dst_tp
       << " and " << src_tp[0];
    throw invalid_argument(ss.str());
  }
  const ndt::type &dst_val_tp =
      dst_tp.extended<ndt::option_type>()->get_value_type();
  const ndt::type &src_val_tp =
      src_tp[0].extended<ndt::option_type>()->get_value_type();
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  // instantiate src_is_avail
  const arrfunc_type_data *af =
      src_tp[0].extended<ndt::option_type>()->get_is_avail().get();
  ckb_offset = af->instantiate(NULL, 0, NULL, ckb, ckb_offset,
                               ndt::make_type<bool1>(), NULL, nsrc, src_tp,
                               src_arrmeta, kernreq, ectx, kwds, tp_vars);
  // instantiate dst_assign_na
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
      ->reserve(ckb_offset + sizeof(ckernel_prefix));
  self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
             ->get_at<self_type>(root_ckb_offset);
  self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
  af = dst_tp.extended<ndt::option_type>()->get_assign_na().get();
  ckb_offset =
      af->instantiate(NULL, 0, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                      NULL, NULL, kernreq, ectx, kwds, tp_vars);
  // instantiate value_assign
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
      ->reserve(ckb_offset + sizeof(ckernel_prefix));
  self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
             ->get_at<self_type>(root_ckb_offset);
  self->m_value_assign_offset = ckb_offset - root_ckb_offset;
  ckb_offset =
      make_assignment_kernel(ckb, ckb_offset, dst_val_tp, dst_arrmeta,
                             src_val_tp, src_arrmeta[0], kernreq, ectx);
  return ckb_offset;
}

static intptr_t instantiate_option_to_value_assignment_kernel(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  intptr_t root_ckb_offset = ckb_offset;
  typedef option_to_value_ck self_type;
  if (dst_tp.get_type_id() == option_type_id ||
      src_tp[0].get_type_id() != option_type_id) {
    stringstream ss;
    ss << "option to value kernel needs value/option types, got " << dst_tp
       << " and " << src_tp[0];
    throw invalid_argument(ss.str());
  }
  const ndt::type &src_val_tp =
      src_tp[0].extended<ndt::option_type>()->get_value_type();
  self_type *self = self_type::make(ckb, kernreq, ckb_offset);
  // instantiate src_is_avail
  const arrfunc_type_data *af =
      src_tp[0].extended<ndt::option_type>()->get_is_avail().get();
  ckb_offset = af->instantiate(NULL, 0, NULL, ckb, ckb_offset,
                               ndt::make_type<bool1>(), NULL, nsrc, src_tp,
                               src_arrmeta, kernreq, ectx, kwds, tp_vars);
  // instantiate value_assign
  reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
      ->reserve(ckb_offset + sizeof(ckernel_prefix));
  self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
             ->get_at<self_type>(root_ckb_offset);
  self->m_value_assign_offset = ckb_offset - root_ckb_offset;
  return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                src_val_tp, src_arrmeta[0], kernreq, ectx);
}

namespace {
struct string_to_option_bool_ck
    : nd::base_kernel<string_to_option_bool_ck, kernel_request_host, 1> {
  assign_error_mode m_errmode;

  void single(char *dst, char *const *src)
  {
    const string_type_data *std = reinterpret_cast<string_type_data *>(src[0]);
    parse::string_to_bool(dst, std->begin, std->end, true, m_errmode);
  }
};

struct string_to_option_number_ck
    : nd::base_kernel<string_to_option_number_ck, kernel_request_host, 1> {
  type_id_t m_tid;
  assign_error_mode m_errmode;

  void single(char *dst, char *const *src)
  {
    const string_type_data *std = reinterpret_cast<string_type_data *>(src[0]);
    parse::string_to_number(dst, m_tid, std->begin, std->end, true, m_errmode);
  }
};

struct string_to_option_tp_ck
    : nd::base_kernel<string_to_option_tp_ck, kernel_request_host, 1> {
  intptr_t m_dst_assign_na_offset;

  void single(char *dst, char *const *src)
  {
    const string_type_data *std = reinterpret_cast<string_type_data *>(src[0]);
    if (parse::matches_option_type_na_token(std->begin, std->end)) {
      // It's not available, assign an NA
      ckernel_prefix *dst_assign_na = get_child_ckernel(m_dst_assign_na_offset);
      expr_single_t dst_assign_na_fn =
          dst_assign_na->get_function<expr_single_t>();
      dst_assign_na_fn(dst, NULL, dst_assign_na);
    } else {
      // It's available, copy using value assignment
      ckernel_prefix *value_assign = get_child_ckernel();
      expr_single_t value_assign_fn =
          value_assign->get_function<expr_single_t>();
      value_assign_fn(dst, src, value_assign);
    }
  }

  void destruct_children()
  {
    // value_assign
    get_child_ckernel()->destroy();
    // dst_assign_na
    destroy_child_ckernel(m_dst_assign_na_offset);
  }
};
}

static intptr_t instantiate_string_to_option_assignment_kernel(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  // Deal with some string to option[T] conversions where string values
  // might mean NA
  if (dst_tp.get_type_id() != option_type_id ||
      !(src_tp[0].get_kind() == string_kind ||
        (src_tp[0].get_type_id() == option_type_id &&
         src_tp[0].extended<ndt::option_type>()->get_value_type().get_kind() ==
             string_kind))) {
    stringstream ss;
    ss << "string to option kernel needs string/option types, got ("
       << src_tp[0] << ") -> " << dst_tp;
    throw invalid_argument(ss.str());
  }

  type_id_t tid =
      dst_tp.extended<ndt::option_type>()->get_value_type().get_type_id();
  switch (tid) {
  case bool_type_id: {
    string_to_option_bool_ck *self =
        string_to_option_bool_ck::make(ckb, kernreq, ckb_offset);
    self->m_errmode = ectx->errmode;
    return ckb_offset;
  }
  case int8_type_id:
  case int16_type_id:
  case int32_type_id:
  case int64_type_id:
  case int128_type_id:
  case float16_type_id:
  case float32_type_id:
  case float64_type_id: {
    string_to_option_number_ck *self =
        string_to_option_number_ck::make(ckb, kernreq, ckb_offset);
    self->m_tid = tid;
    self->m_errmode = ectx->errmode;
    return ckb_offset;
  }
  case string_type_id: {
    // Just a string to string assignment
    return ::make_assignment_kernel(
        ckb, ckb_offset, dst_tp.extended<ndt::option_type>()->get_value_type(),
        dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
  }
  default:
    break;
  }

  // Fall back to an adaptor that checks for a few standard
  // missing value tokens, then uses the standard value assignment
  intptr_t root_ckb_offset = ckb_offset;
  string_to_option_tp_ck *self =
      string_to_option_tp_ck::make(ckb, kernreq, ckb_offset);
  // First child ckernel is the value assignment
  ckb_offset = ::make_assignment_kernel(
      ckb, ckb_offset, dst_tp.extended<ndt::option_type>()->get_value_type(),
      dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
  // Re-acquire self because the address may have changed
  self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
             ->get_at<string_to_option_tp_ck>(root_ckb_offset);
  // Second child ckernel is the NA assignment
  self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
  const arrfunc_type_data *af =
      dst_tp.extended<ndt::option_type>()->get_assign_na().get();
  ckb_offset =
      af->instantiate(NULL, 0, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                      NULL, NULL, kernreq, ectx, kwds, tp_vars);
  return ckb_offset;
}

static intptr_t instantiate_float_to_option_assignment_kernel(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  // Deal with some float32 to option[T] conversions where any NaN is
  // interpreted
  // as NA.
  ndt::type src_tp_as_option = ndt::make_option(src_tp[0]);
  return instantiate_option_to_option_assignment_kernel(
      NULL, 0, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
      &src_tp_as_option, src_arrmeta, kernreq, ectx, kwds, tp_vars);
}

static intptr_t instantiate_option_as_value_assignment_kernel(
    char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &DYND_UNUSED(kwds),
    const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  // In all cases not handled, we use the
  // regular S to T assignment kernel.
  //
  // Note that this does NOT catch the case where a value
  // which was ok with type S, but equals the NA
  // token in type T, is assigned. Checking this
  // properly across all the cases would add
  // fairly significant cost, and it seems maybe ok
  // to skip it.
  ndt::type val_dst_tp =
      dst_tp.get_type_id() == option_type_id
          ? dst_tp.extended<ndt::option_type>()->get_value_type()
          : dst_tp;
  ndt::type val_src_tp =
      src_tp[0].get_type_id() == option_type_id
          ? src_tp[0].extended<ndt::option_type>()->get_value_type()
          : src_tp[0];
  return ::make_assignment_kernel(ckb, ckb_offset, val_dst_tp, dst_arrmeta,
                                  val_src_tp, src_arrmeta[0], kernreq, ectx);
}

namespace {

struct option_arrfunc_list {
  ndt::type af_tp[7];
  arrfunc_type_data af[7];

  option_arrfunc_list()
  {
    int i = 0;
    af_tp[i] = ndt::type("(?string) -> ?S");
    af[i].instantiate = &instantiate_string_to_option_assignment_kernel;
    ++i;
    af_tp[i] = ndt::type("(?T) -> ?S");
    af[i].instantiate = &instantiate_option_to_option_assignment_kernel;
    ++i;
    af_tp[i] = ndt::type("(?T) -> S");
    af[i].instantiate = &instantiate_option_to_value_assignment_kernel;
    ++i;
    af_tp[i] = ndt::type("(string) -> ?S");
    af[i].instantiate = &instantiate_string_to_option_assignment_kernel;
    ++i;
    af_tp[i] = ndt::type("(float32) -> ?S");
    af[i].instantiate = &instantiate_float_to_option_assignment_kernel;
    ++i;
    af_tp[i] = ndt::type("(float64) -> ?S");
    af[i].instantiate = &instantiate_float_to_option_assignment_kernel;
    ++i;
    af_tp[i] = ndt::type("(T) -> S");
    af[i].instantiate = &instantiate_option_as_value_assignment_kernel;
  }

  inline intptr_t size() const { return sizeof(af) / sizeof(af[0]); }

  const arrfunc_type_data *get() const { return af; }
  const ndt::type *get_type() const { return af_tp; }
};
} // anonymous namespace

size_t kernels::make_option_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  static option_arrfunc_list afl;
  intptr_t size = afl.size();
  const arrfunc_type_data *af = afl.get();
  const ndt::arrfunc_type *const *af_tp =
      reinterpret_cast<const ndt::arrfunc_type *const *>(afl.get_type());
  map<nd::string, ndt::type> typevars;
  for (intptr_t i = 0; i < size; ++i, ++af_tp, ++af) {
    typevars.clear();
    if ((*af_tp)->get_pos_type(0).match(src_tp, typevars) &&
        (*af_tp)->get_return_type().match(dst_tp, typevars)) {
      return af->instantiate(NULL, 0, NULL, ckb, ckb_offset, dst_tp,
                             dst_arrmeta, size, &src_tp, &src_arrmeta, kernreq,
                             ectx, nd::array(),
                             std::map<nd::string, ndt::type>());
    }
  }

  stringstream ss;
  ss << "Could not instantiate option assignment kernel from " << src_tp
     << " to " << dst_tp;
  throw invalid_argument(ss.str());
}
