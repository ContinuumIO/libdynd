//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/reduce.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/asarray.hpp>

using namespace std;
using namespace dynd;

namespace {

/**
 * Reduction structure is built with two child functions per nested ckernel
 * instead of one as in most cases. This serves the same purpose as the
 * `FirstVisit` property in NumPy's nditer
 * (http://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#c.NpyIter_IsFirstVisit),
 * where this knowledge allows a reduction loop to choose between initialization
 * and accumulation as necessary.
 *
 * There are two ways a one-dimensional reduction will typically work, depending
 * on whether the operation in question has an identity or not. With an
 * identity, like '+' has 0, we get steps as follows for example [1, 3, 5]:
 *   - output is uninitialized (?)
 *   - do single first_call, with no input value, output becomes (0)
 *   - do strided followup_call, which does a strided loop adding 1, 3, then 5
 *   - output is now (9)
 *
 * Without an identity, like 'max', we get steps as follows for the example:
 *   - output is uninitialized (?)
 *   - do single first_call, with the first input (1), output becomes (1)
 *   - do strided followup_call on the rest, strided loop maxing 3, then 5
 *   - output is now (5)
 *
 * This plays out in a nested fashion, where each dimension may be reduced or
 * not, depending on input axis selection. Additionally, in the second case, by
 * allowing a general first_call accumulator initialization instead of
 * hardcoding it as a copy, this enables the expression of more general
 * accumulators that might accumulate multiple statistical properties at once.
 */
struct reduction_ckernel_prefix : ckernel_prefix {
  // This function pointer is for all the calls of the function
  // on a given destination data address after the "first call".
  expr_strided_t followup_call_function;

  template <typename T>
  T get_first_call_function() const
  {
    return get_function<T>();
  }

  template <typename T>
  void set_first_call_function(T fnptr)
  {
    set_function<T>(fnptr);
  }

  expr_strided_t get_followup_call_function() const
  {
    return followup_call_function;
  }

  void set_followup_call_function(expr_strided_t fnptr)
  {
    followup_call_function = fnptr;
  }
};

/**
 * STRIDED INITIAL REDUCTION DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a reduction dimension, so dst_stride is zero.
 *  - It's an initial dimension, there are additional dimensions
 *    being processed by its child kernels.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child first_call function must be *single*.
 *  - The child followup_call function must be *strided*.
 *
 */
struct strided_initial_reduction_kernel_extra
    : nd::base_kernel<strided_initial_reduction_kernel_extra,
                      kernel_request_host, 1, reduction_ckernel_prefix> {
  typedef strided_initial_reduction_kernel_extra self_type;

  // The code assumes that size >= 1
  intptr_t size;
  intptr_t src_stride;

  static void single_first(char *dst, char *const *src, ckernel_prefix *rawself)
  {
    self_type *e = get_self(rawself);
    reduction_ckernel_prefix *echild =
        reinterpret_cast<reduction_ckernel_prefix *>(e->get_child_ckernel());
    // The first call at the "dst" address
    expr_single_t opchild_first_call =
        echild->get_first_call_function<expr_single_t>();
    opchild_first_call(dst, src, echild);
    if (e->size > 1) {
      // All the followup calls at the "dst" address
      expr_strided_t opchild = echild->get_followup_call_function();
      char *src_second = src[0] + e->src_stride;
      opchild(dst, 0, &src_second, &e->src_stride, e->size - 1, echild);
    }
  }

  static void strided_first(char *dst, intptr_t dst_stride, char *const *src,
                            const intptr_t *src_stride, size_t count,
                            ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    reduction_ckernel_prefix *echild =
        reinterpret_cast<reduction_ckernel_prefix *>(e->get_child_ckernel());
    expr_strided_t opchild_followup_call = echild->get_followup_call_function();
    expr_single_t opchild_first_call =
        echild->get_first_call_function<expr_single_t>();
    intptr_t inner_size = e->size;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      // With a zero stride, we have one "first", followed by many "followup"
      // calls
      opchild_first_call(dst, &src0, echild);
      if (inner_size > 1) {
        char *inner_src_second = src0 + inner_src_stride;
        opchild_followup_call(dst, 0, &inner_src_second, &inner_src_stride,
                              inner_size - 1, echild);
      }
      src0 += src0_stride;
      for (intptr_t i = 1; i < (intptr_t)count; ++i) {
        opchild_followup_call(dst, 0, &src0, &inner_src_stride, inner_size,
                              echild);
        src0 += src0_stride;
      }
    } else {
      // With a non-zero stride, each iteration of the outer loop is "first"
      for (size_t i = 0; i != count; ++i) {
        opchild_first_call(dst, &src0, echild);
        if (inner_size > 1) {
          char *inner_src_second = src0 + inner_src_stride;
          opchild_followup_call(dst, 0, &inner_src_second, &inner_src_stride,
                                inner_size - 1, echild);
        }
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }

  static void strided_followup(char *dst, intptr_t dst_stride, char *const *src,
                               const intptr_t *src_stride, size_t count,
                               ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    reduction_ckernel_prefix *echild =
        reinterpret_cast<reduction_ckernel_prefix *>(e->get_child_ckernel());
    expr_strided_t opchild_followup_call = echild->get_followup_call_function();
    intptr_t inner_size = e->size;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      opchild_followup_call(dst, 0, &src0, &inner_src_stride, inner_size,
                            echild);
      dst += dst_stride;
      src0 += src0_stride;
    }
  }

  void destruct_children()
  {
    get_child_ckernel()->destroy();
  }

  /**
   * Adds a ckernel layer for processing one dimension of the reduction.
   * This is for a strided dimension which is being reduced, and is not
   * the final dimension before the accumulation operation.
   */
  static size_t make(void *ckb, intptr_t ckb_offset, intptr_t src_stride,
                     intptr_t src_dim_size, kernel_request_t kernreq)
  {
    self_type *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                       ->alloc_ck<self_type>(ckb_offset);
    e->destructor = &self_type::destruct;
    // Get the function pointer for the first_call
    if (kernreq == kernel_request_single) {
      e->set_first_call_function(&self_type::single_first);
    }
    else if (kernreq == kernel_request_strided) {
      e->set_first_call_function(&self_type::strided_first);
    }
    else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
    // The function pointer for followup accumulation calls
    e->set_followup_call_function(&self_type::strided_followup);
    // The striding parameters
    e->src_stride = src_stride;
    e->size = src_dim_size;
    return ckb_offset;
  }
};

/**
 * STRIDED INITIAL BROADCAST DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a broadcast dimension, so dst_stride is not zero.
 *  - It's an initial dimension, there are additional dimensions
 *    being processed after this one.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child first_call function must be *strided*.
 *  - The child followup_call function must be *strided*.
 *
 */
struct strided_initial_broadcast_kernel_extra
    : nd::base_kernel<strided_initial_broadcast_kernel_extra,
                      kernel_request_host, 1, reduction_ckernel_prefix> {
  typedef strided_initial_broadcast_kernel_extra self_type;

  // The code assumes that size >= 1
  intptr_t size;
  intptr_t dst_stride, src_stride;

  static void single_first(char *dst, char *const *src, ckernel_prefix *rawself)
  {
    self_type *e = get_self(rawself);
    reduction_ckernel_prefix *echild =
        reinterpret_cast<reduction_ckernel_prefix *>(e->get_child_ckernel());
    expr_strided_t opchild_first_call =
        echild->get_first_call_function<expr_strided_t>();
    opchild_first_call(dst, e->dst_stride, src, &e->src_stride, e->size,
                       echild);
  }

  static void strided_first(char *dst, intptr_t dst_stride, char *const *src,
                            const intptr_t *src_stride, size_t count,
                            ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    reduction_ckernel_prefix *echild =
        reinterpret_cast<reduction_ckernel_prefix *>(e->get_child_ckernel());
    expr_strided_t opchild_first_call =
        echild->get_first_call_function<expr_strided_t>();
    expr_strided_t opchild_followup_call = echild->get_followup_call_function();
    intptr_t inner_size = e->size;
    intptr_t inner_dst_stride = e->dst_stride;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      // With a zero stride, we have one "first", followed by many "followup"
      // calls
      opchild_first_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                         inner_size, echild);
      dst += dst_stride;
      src0 += src0_stride;
      for (intptr_t i = 1; i < (intptr_t)count; ++i) {
        opchild_followup_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                              inner_size, echild);
        dst += dst_stride;
        src0 += src0_stride;
      }
    } else {
      // With a non-zero stride, each iteration of the outer loop is "first"
      for (size_t i = 0; i != count; ++i) {
        opchild_first_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                           inner_size, echild);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }

  static void strided_followup(char *dst, intptr_t dst_stride, char *const *src,
                               const intptr_t *src_stride, size_t count,
                               ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    reduction_ckernel_prefix *echild =
        reinterpret_cast<reduction_ckernel_prefix *>(e->get_child_ckernel());
    expr_strided_t opchild_followup_call = echild->get_followup_call_function();
    intptr_t inner_size = e->size;
    intptr_t inner_dst_stride = e->dst_stride;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      opchild_followup_call(dst, inner_dst_stride, &src0, &inner_src_stride,
                            inner_size, echild);
      dst += dst_stride;
      src0 += src0_stride;
    }
  }

  void destruct_children()
  {
    get_child_ckernel()->destroy();
  }

  /**
   * Adds a ckernel layer for processing one dimension of the reduction.
   * This is for a strided dimension which is being broadcast, and is not
   * the final dimension before the accumulation operation.
   */
  static size_t make(void *ckb, intptr_t ckb_offset, intptr_t dst_stride,
                     intptr_t src_stride, intptr_t src_dim_size,
                     kernel_request_t kernreq)
  {
    self_type *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                       ->alloc_ck<self_type>(ckb_offset);
    e->destructor = &self_type::destruct;
    // Get the function pointer for the first_call
    if (kernreq == kernel_request_single) {
      e->set_first_call_function(&self_type::single_first);
    }
    else if (kernreq == kernel_request_strided) {
      e->set_first_call_function(&self_type::strided_first);
    }
    else {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: unrecognized request "
         << (int)kernreq;
      throw runtime_error(ss.str());
    }
    // The function pointer for followup accumulation calls
    e->set_followup_call_function(&self_type::strided_followup);
    // The striding parameters
    e->dst_stride = dst_stride;
    e->src_stride = src_stride;
    e->size = src_dim_size;
    return ckb_offset;
  }
};

static void check_dst_initialization(const ndt::arrfunc_type *dst_initialization_tp,
                                     const ndt::type &dst_tp,
                                     const ndt::type &src_tp)
{
  if (dst_initialization_tp->get_return_type() != dst_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
    ss << "dst type is " << dst_initialization_tp->get_return_type();
    ss << ", expected " << dst_tp;
    throw type_error(ss.str());
  }
  if (dst_initialization_tp->get_pos_type(0) != src_tp) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: dst initialization ckernel ";
    ss << "src type is " << dst_initialization_tp->get_return_type();
    ss << ", expected " << src_tp;
    throw type_error(ss.str());
  }
}

/**
 * STRIDED INNER REDUCTION DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a reduction dimension, so dst_stride is zero.
 *  - It's an inner dimension, calling the reduction kernel directly.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child destination initialization kernel must be *single*.
 *  - The child reduction kernel must be *strided*.
 *
 */
struct strided_inner_reduction_kernel_extra
    : nd::base_kernel<strided_inner_reduction_kernel_extra, kernel_request_host,
                      1, reduction_ckernel_prefix> {
  typedef strided_inner_reduction_kernel_extra self_type;

  // The code assumes that size >= 1
  intptr_t size;
  intptr_t src_stride;
  size_t dst_init_kernel_offset;
  // For the case with a reduction identity
  const char *ident_data;
  memory_block_data *ident_ref;

  static void single_first(char *dst, char *const *src, ckernel_prefix *rawself)
  {
    self_type *e = get_self(rawself);
    ckernel_prefix *echild_reduce = e->get_child_ckernel();
    ckernel_prefix *echild_dst_init =
        e->get_child_ckernel(e->dst_init_kernel_offset);
    // The first call to initialize the "dst" value
    expr_single_t opchild_dst_init =
        echild_dst_init->get_function<expr_single_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    opchild_dst_init(dst, src, echild_dst_init);
    if (e->size > 1) {
      // All the followup calls to accumulate at the "dst" address
      char *child_src = src[0] + e->src_stride;
      opchild_reduce(dst, 0, &child_src, &e->src_stride, e->size - 1,
                     echild_reduce);
    }
  }

  static void single_first_with_ident(char *dst, char *const *src,
                                      ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
    // The first call to initialize the "dst" value
    expr_single_t opchild_ident = echild_ident->get_function<expr_single_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    opchild_ident(dst, const_cast<char *const *>(&e->ident_data), echild_ident);
    // All the followup calls to accumulate at the "dst" address
    opchild_reduce(dst, 0, src, &e->src_stride, e->size, echild_reduce);
  }

  static void strided_first(char *dst, intptr_t dst_stride, char *const *src,
                            const intptr_t *src_stride, size_t count,
                            ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
    expr_single_t opchild_dst_init =
        echild_dst_init->get_function<expr_single_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    intptr_t inner_size = e->size;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      // With a zero stride, we initialize "dst" once, then do many
      // accumulations
      opchild_dst_init(dst, &src0, echild_dst_init);
      if (inner_size > 1) {
        char *inner_child_src = src0 + inner_src_stride;
        opchild_reduce(dst, 0, &inner_child_src, &inner_src_stride,
                       inner_size - 1, echild_reduce);
      }
      src0 += src0_stride;
      for (intptr_t i = 1; i < (intptr_t)count; ++i) {
        opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                       echild_reduce);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
    else {
      // With a non-zero stride, each iteration of the outer loop has to
      // initialize then reduce
      for (size_t i = 0; i != count; ++i) {
        opchild_dst_init(dst, &src0, echild_dst_init);
        if (inner_size > 1) {
          char *inner_child_src = src0 + inner_src_stride;
          opchild_reduce(dst, 0, &inner_child_src, &inner_src_stride,
                         inner_size - 1, echild_reduce);
        }
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }

  static void strided_first_with_ident(char *dst, intptr_t dst_stride,
                                       char *const *src,
                                       const intptr_t *src_stride, size_t count,
                                       ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
    expr_single_t opchild_ident = echild_ident->get_function<expr_single_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    const char *ident_data = e->ident_data;
    intptr_t inner_size = e->size;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      // With a zero stride, we initialize "dst" once, then do many
      // accumulations
      opchild_ident(dst, const_cast<char *const *>(&ident_data), echild_ident);
      for (intptr_t i = 0; i < (intptr_t)count; ++i) {
        opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                       echild_reduce);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
    else {
      // With a non-zero stride, each iteration of the outer loop has to
      // initialize then reduce
      for (size_t i = 0; i != count; ++i) {
        opchild_ident(dst, const_cast<char *const *>(&ident_data),
                      echild_ident);
        opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                       echild_reduce);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }

  static void strided_followup(char *dst, intptr_t dst_stride, char *const *src,
                               const intptr_t *src_stride, size_t count,
                               ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    // No initialization, all reduction
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    intptr_t inner_size = e->size;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      opchild_reduce(dst, 0, &src0, &inner_src_stride, inner_size,
                     echild_reduce);
      dst += dst_stride;
      src0 += src0_stride;
    }
  }

  void destruct_children()
  {
    if (ident_ref != NULL) {
      memory_block_decref(ident_ref);
    }
    // The reduction kernel
    get_child_ckernel()->destroy();
    // The destination initialization kernel
    destroy_child_ckernel(dst_init_kernel_offset);
  }

  /**
   * Adds a ckernel layer for processing one dimension of the reduction.
   * This is for a strided dimension which is being reduced, and is
   * the final dimension before the accumulation operation.
   *
   * If dst_init and red_ident are both NULL, an assignment kernel is used.
   */
  static size_t make(const nd::arrfunc &red_op, const nd::arrfunc &dst_init,
                     void *ckb, intptr_t ckb_offset, intptr_t src_stride,
                     intptr_t src_dim_size, const ndt::type &dst_tp,
                     const char *dst_arrmeta, const ndt::type &src_tp,
                     const char *src_arrmeta, bool right_associative,
                     const nd::array &red_ident,
                     kernel_request_t kernreq, const eval::eval_context *ectx)
  {
    intptr_t root_ckb_offset = ckb_offset;
    self_type *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                       ->alloc_ck<self_type>(ckb_offset);
    e->destructor = &self_type::destruct;
    // Cannot have both a dst_init kernel and a reduction identity
    if (!dst_init.is_null() && !red_ident.is_null()) {
      throw invalid_argument(
          "make_lifted_reduction_ckernel: cannot specify"
          " both a dst_init kernel and a red_ident");
    }
    if (red_ident.is_null()) {
      // Get the function pointer for the first_call, for the case with
      // no reduction identity
      if (kernreq == kernel_request_single) {
        e->set_first_call_function(&self_type::single_first);
      }
      else if (kernreq == kernel_request_strided) {
        e->set_first_call_function(&self_type::strided_first);
      }
      else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request "
           << (int)kernreq;
        throw runtime_error(ss.str());
      }
    }
    else {
      // Get the function pointer for the first_call, for the case with
      // a reduction identity
      if (kernreq == kernel_request_single) {
        e->set_first_call_function(&self_type::single_first_with_ident);
      }
      else if (kernreq == kernel_request_strided) {
        e->set_first_call_function(&self_type::strided_first_with_ident);
      }
      else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request "
           << (int)kernreq;
        throw runtime_error(ss.str());
      }
      if (red_ident.get_type() != dst_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: reduction identity type ";
        ss << red_ident.get_type() << " does not match dst type ";
        ss << dst_tp;
        throw runtime_error(ss.str());
      }
      e->ident_data = red_ident.get_readonly_originptr();
      e->ident_ref = red_ident.get_memblock().release();
    }
    // The function pointer for followup accumulation calls
    e->set_followup_call_function(&self_type::strided_followup);
    // The striding parameters
    e->src_stride = src_stride;
    e->size = src_dim_size;
    // Validate that the provided arrfuncs are unary operations,
    // and have the correct types
    if (red_op.get_npos() != 1 &&
        red_op.get_npos() != 2) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
      ss << "funcproto must be unary or a binary expr with all equal types";
      throw runtime_error(ss.str());
    }
    if (red_op.get_return_type() != dst_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
      ss << "dst type is " << red_op.get_return_type();
      ss << ", expected " << dst_tp;
      throw type_error(ss.str());
    }
    if (red_op.get_pos_type(0) != src_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
      ss << "src type is " << red_op.get_return_type();
      ss << ", expected " << src_tp;
      throw type_error(ss.str());
    }
    if (!dst_init.is_null()) {
      check_dst_initialization(dst_init.get_type(), dst_tp, src_tp);
    }
    if (red_op.get_npos() == 2) {
      ckb_offset = kernels::wrap_binary_as_unary_reduction_ckernel(
          ckb, ckb_offset, right_associative, kernel_request_strided);
      ndt::type src_tp_doubled[2] = {src_tp, src_tp};
      const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
      ckb_offset = red_op.get()->instantiate(
          red_op.get(), red_op.get_type(), NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, red_op.get_npos(), src_tp_doubled, src_arrmeta_doubled,
          kernel_request_strided, ectx, nd::array(),
          std::map<nd::string, ndt::type>());
    }
    else {
      ckb_offset = red_op.get()->instantiate(
          red_op.get(), red_op.get_type(), NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, 1, &src_tp, &src_arrmeta, kernel_request_strided, ectx,
          nd::array(), std::map<nd::string, ndt::type>());
    }
    // Make sure there's capacity for the next ckernel
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->reserve(ckb_offset + sizeof(ckernel_prefix));
    // Need to retrieve 'e' again because it may have moved
    e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->get_at<self_type>(root_ckb_offset);
    e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
    if (!dst_init.is_null()) {
      ckb_offset = dst_init.get()->instantiate(
          dst_init.get(), dst_init.get_type(), NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, 1, &src_tp, &src_arrmeta, kernel_request_single,
          ectx, nd::array(), std::map<nd::string, ndt::type>());
    }
    else if (red_ident.is_null()) {
      ckb_offset = make_assignment_kernel(
          NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
          kernel_request_single, ectx, nd::array());
    }
    else {
      ckb_offset = make_assignment_kernel(
          NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
          red_ident.get_type(), red_ident.get_arrmeta(),
          kernel_request_single, ectx, nd::array());
    }

    return ckb_offset;
  }
};

/**
 * STRIDED INNER BROADCAST DIMENSION
 * This ckernel handles one dimension of the reduction processing,
 * where:
 *  - It's a broadcast dimension, so dst_stride is not zero.
 *  - It's an inner dimension, calling the reduction kernel directly.
 *  - The source data is strided.
 *
 * Requirements:
 *  - The child reduction kernel must be *strided*.
 *  - The child destination initialization kernel must be *strided*.
 *
 */
struct strided_inner_broadcast_kernel_extra
    : nd::base_kernel<strided_inner_broadcast_kernel_extra, kernel_request_host,
                      1, reduction_ckernel_prefix> {
  typedef strided_inner_broadcast_kernel_extra self_type;

  // The code assumes that size >= 1
  intptr_t size;
  intptr_t dst_stride, src_stride;
  size_t dst_init_kernel_offset;
  // For the case with a reduction identity
  const char *ident_data;
  memory_block_data *ident_ref;

  static void single_first(char *dst, char *const *src, ckernel_prefix *rawself)
  {
    self_type *e = get_self(rawself);
    ckernel_prefix *echild_dst_init =
        e->get_child_ckernel(e->dst_init_kernel_offset);
    expr_strided_t opchild_dst_init =
        echild_dst_init->get_function<expr_strided_t>();
    // All we do is initialize the dst values
    opchild_dst_init(dst, e->dst_stride, src, &e->src_stride, e->size,
                     echild_dst_init);
  }

  static void single_first_with_ident(char *dst, char *const *src,
                                      ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    expr_strided_t opchild_ident = echild_ident->get_function<expr_strided_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    // First initialize the dst values (TODO: Do we want to do initialize/reduce
    // in blocks instead of just one then the other?)
    intptr_t zero_stride = 0;
    opchild_ident(dst, e->dst_stride, const_cast<char *const *>(&e->ident_data),
                  &zero_stride, e->size, echild_ident);
    // Then do the accumulation
    opchild_reduce(dst, e->dst_stride, src, &e->src_stride, e->size,
                   echild_reduce);
  }

  static void strided_first(char *dst, intptr_t dst_stride, char *const *src,
                            const intptr_t *src_stride, size_t count,
                            ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_dst_init = reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    expr_strided_t opchild_dst_init =
        echild_dst_init->get_function<expr_strided_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    intptr_t inner_size = e->size;
    intptr_t inner_dst_stride = e->dst_stride;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      // With a zero stride, we initialize "dst" once, then do many
      // accumulations
      opchild_dst_init(dst, inner_dst_stride, &src0, &inner_src_stride,
                       inner_size, echild_dst_init);
      dst += dst_stride;
      src0 += src0_stride;
      for (intptr_t i = 1; i < (intptr_t)count; ++i) {
        opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                       inner_size, echild_reduce);
        src0 += src0_stride;
      }
    }
    else {
      // With a non-zero stride, every iteration is an initialization
      for (size_t i = 0; i != count; ++i) {
        opchild_dst_init(dst, inner_dst_stride, &src0, &inner_src_stride,
                         e->size, echild_dst_init);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }

  static void strided_first_with_ident(char *dst, intptr_t dst_stride,
                                       char *const *src,
                                       const intptr_t *src_stride, size_t count,
                                       ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_ident = reinterpret_cast<ckernel_prefix *>(
        reinterpret_cast<char *>(extra) + e->dst_init_kernel_offset);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    expr_strided_t opchild_ident = echild_ident->get_function<expr_strided_t>();
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    intptr_t inner_size = e->size;
    intptr_t inner_dst_stride = e->dst_stride;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    if (dst_stride == 0) {
      // With a zero stride, we initialize "dst" once, then do many
      // accumulations
      intptr_t zero_stride = 0;
      opchild_ident(dst, inner_dst_stride,
                    const_cast<char *const *>(&e->ident_data), &zero_stride,
                    e->size, echild_ident);
      for (intptr_t i = 0; i < (intptr_t)count; ++i) {
        opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                       inner_size, echild_reduce);
        src0 += src0_stride;
      }
    }
    else {
      intptr_t zero_stride = 0;
      // With a non-zero stride, every iteration is an initialization
      for (size_t i = 0; i != count; ++i) {
        opchild_ident(dst, inner_dst_stride,
                      const_cast<char *const *>(&e->ident_data), &zero_stride,
                      inner_size, echild_ident);
        opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                       inner_size, echild_reduce);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  }

  static void strided_followup(char *dst, intptr_t dst_stride, char *const *src,
                               const intptr_t *src_stride, size_t count,
                               ckernel_prefix *extra)
  {
    self_type *e = reinterpret_cast<self_type *>(extra);
    ckernel_prefix *echild_reduce = extra->get_child_ckernel(sizeof(self_type));
    // No initialization, all reduction
    expr_strided_t opchild_reduce =
        echild_reduce->get_function<expr_strided_t>();
    intptr_t inner_size = e->size;
    intptr_t inner_dst_stride = e->dst_stride;
    intptr_t inner_src_stride = e->src_stride;
    char *src0 = src[0];
    intptr_t src0_stride = src_stride[0];
    for (size_t i = 0; i != count; ++i) {
      opchild_reduce(dst, inner_dst_stride, &src0, &inner_src_stride,
                     inner_size, echild_reduce);
      dst += dst_stride;
      src0 += src0_stride;
    }
  }

  void destruct_children()
  {
    if (ident_ref != NULL) {
      memory_block_decref(ident_ref);
    }
    // The reduction kernel
    get_child_ckernel()->destroy();
    // The destination initialization kernel
    destroy_child_ckernel(dst_init_kernel_offset);
  }

  /**
   * Adds a ckernel layer for processing one dimension of the reduction.
   * This is for a strided dimension which is being broadcast, and is
   * the final dimension before the accumulation operation.
   */
  static size_t make(const nd::arrfunc &red_op, const nd::arrfunc &dst_init,
                     void *ckb, intptr_t ckb_offset, intptr_t dst_stride,
                     intptr_t src_stride, intptr_t src_dim_size,
                     const ndt::type &dst_tp, const char *dst_arrmeta,
                     const ndt::type &src_tp, const char *src_arrmeta,
                     bool right_associative, const nd::array &red_ident,
                     kernel_request_t kernreq, const eval::eval_context *ectx)
  {
    intptr_t root_ckb_offset = ckb_offset;
    self_type *e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                       ->alloc_ck<self_type>(ckb_offset);
    e->destructor = &self_type::destruct;
    // Cannot have both a dst_init kernel and a reduction identity
    if (!dst_init.is_null() && !red_ident.is_null()) {
      throw invalid_argument("make_lifted_reduction_ckernel: cannot specify "
                             "both a dst_init kernel and a red_ident");
    }
    if (red_ident.is_null()) {
      // Get the function pointer for the first_call, for the case with
      // no reduction identity
      if (kernreq == kernel_request_single) {
        e->set_first_call_function(&self_type::single_first);
      }
      else if (kernreq == kernel_request_strided) {
        e->set_first_call_function(&self_type::strided_first);
      }
      else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request "
           << (int)kernreq;
        throw runtime_error(ss.str());
      }
    }
    else {
      // Get the function pointer for the first_call, for the case with
      // a reduction identity
      if (kernreq == kernel_request_single) {
        e->set_first_call_function(&self_type::single_first_with_ident);
      }
      else if (kernreq == kernel_request_strided) {
        e->set_first_call_function(&self_type::strided_first_with_ident);
      }
      else {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: unrecognized request "
           << (int)kernreq;
        throw runtime_error(ss.str());
      }
      if (red_ident.get_type() != dst_tp) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: reduction identity type ";
        ss << red_ident.get_type() << " does not match dst type ";
        ss << dst_tp;
        throw runtime_error(ss.str());
      }
      e->ident_data = red_ident.get_readonly_originptr();
      e->ident_ref = red_ident.get_memblock().release();
    }

    // The function pointer for followup accumulation calls
    e->set_followup_call_function(&self_type::strided_followup);
    // The striding parameters
    e->dst_stride = dst_stride;
    e->src_stride = src_stride;
    e->size = src_dim_size;
    // Validate that the provided arrfuncs are unary operations,
    // and have the correct types
    if (red_op.get_npos() != 1 && red_op.get_npos() != 2) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
      ss << "funcproto must be unary or a binary expr with all equal types";
      throw runtime_error(ss.str());
    }
    if (red_op.get_return_type() != dst_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
      ss << "dst type is " << red_op.get_return_type();
      ss << ", expected " << dst_tp;
      throw type_error(ss.str());
    }
    if (red_op.get_pos_type(0) != src_tp) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: elwise reduction ckernel ";
      ss << "src type is " << red_op.get_return_type();
      ss << ", expected " << src_tp;
      throw type_error(ss.str());
    }
    if (!dst_init.is_null()) {
      check_dst_initialization(dst_init.get_type(), dst_tp, src_tp);
    }
    if (red_op.get_npos() == 2) {
      ckb_offset = kernels::wrap_binary_as_unary_reduction_ckernel(
          ckb, ckb_offset, right_associative, kernel_request_strided);
      ndt::type src_tp_doubled[2] = {src_tp, src_tp};
      const char *src_arrmeta_doubled[2] = {src_arrmeta, src_arrmeta};
      ckb_offset = red_op.get()->instantiate(
          red_op.get(), red_op.get_type(), NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, red_op.get_npos(), src_tp_doubled, src_arrmeta_doubled,
          kernel_request_strided, ectx, nd::array(),
          std::map<nd::string, ndt::type>());
    }
    else {
      ckb_offset = red_op.get()->instantiate(
          red_op.get(), red_op.get_type(), NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, red_op.get_npos(), &src_tp, &src_arrmeta,
          kernel_request_strided, ectx, nd::array(),
          std::map<nd::string, ndt::type>());
    }
    // Make sure there's capacity for the next ckernel
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->reserve(ckb_offset + sizeof(ckernel_prefix));
    // Need to retrieve 'e' again because it may have moved
    e = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->get_at<self_type>(root_ckb_offset);
    e->dst_init_kernel_offset = ckb_offset - root_ckb_offset;
    if (!dst_init.is_null()) {
      ckb_offset = dst_init.get()->instantiate(
          dst_init.get(), dst_init.get_type(), NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, red_op.get_npos(), &src_tp, &src_arrmeta,
          kernel_request_strided, ectx, nd::array(),
          std::map<nd::string, ndt::type>());
    }
    else if (red_ident.is_null()) {
      ckb_offset = make_assignment_kernel(
          NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp, src_arrmeta,
          kernel_request_strided, ectx, nd::array());
    }
    else {
      ckb_offset = make_assignment_kernel(
          NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta,
          red_ident.get_type(), red_ident.get_arrmeta(),
          kernel_request_strided, ectx, nd::array());
    }

    return ckb_offset;
  }
};

} // anonymous namespace

void nd::functional::reduce_virtual_ck::resolve_option_values(
    const arrfunc_type_data *self,
    const ndt::arrfunc_type *DYND_UNUSED(self_tp), char *resolution_data,
    intptr_t nsrc, const ndt::type *src_tp, nd::array &kwds,
    const std::map<nd::string, ndt::type> &tp_vars)
{
  // Initialize the resolution data memory
  auto kw = new (resolution_data) resolution_data_type;

  // Axis must be a boolean array with the dimensions being reduced
  nd::array axis = kwds.p("axis");
  intptr_t reduction_ndim = axis.get_dim_size();
  auto reduction_dimflags =
      reinterpret_cast<const dynd_bool *>(axis.get_readonly_originptr());
  // The elwise reduction operation
  nd::arrfunc red_op = kwds.p("op");
  // Function to initialize destination elements, may be NULL
  nd::arrfunc dst_init = kwds.p("dst_init");
  // The reduction identity, may be NULL
  nd::array red_ident = kwds.p("red_ident");
  bool right_associative = kwds.p("right_associative").as<bool>();
  bool associative = kwds.p("associative").as<bool>();
  bool commutative = kwds.p("commutative").as<bool>();
}

size_t nd::functional::reduce_virtual_ck::instantiate(
    const arrfunc_type_data *self, const ndt::arrfunc_type *self_tp,
    char *resolution_data, void *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &DYND_UNUSED(kwds),
    const std::map<dynd::nd::string, ndt::type> &tp_vars)
{
  // Get the kwargs
  auto kw = reinterpret_cast<const resolution_data_type *>(resolution_data);

  // Convert the `axis` input into an array of booleans
  intptr_t reduction_ndim = src_tp[0].get_ndim() - dst_tp.get_ndim();
  shortvector<bool> reduction_dimflags(reduction_ndim);
  if (kw->axis.is_null) {
    // No axis specified means reduce all dimensions
    for (intptr_t i = 0; i < reduction_ndim; ++i) {
      reduction_dimflags[i] = true;
    }
  }
  else {
    // Experimenting with the pieces that could be shaped into a
    // pattern-matching idiom of some kind. The biggest trouble in forming this
    // into something relatively simple to express is the desire to have all the
    // types being matched be statically constructed, not constructed at match
    // time.
    static ndt::type int_kind_tp("Int");
    static ndt::type int_kind_array_tp("Fixed * Int");
    static ndt::type bool_kind_array_tp("Fixed * Bool");

    if (int_kind_tp.match(kw->axis.get_type())) {
      // A single axis
      for (intptr_t i = 0; i < reduction_ndim; ++i) {
        reduction_dimflags[i] = false;
      }
      // TODO: Better out-of-bounds error message referencing that it's a
      //       dimension that is out of bounds
      intptr_t dim = kw->axis.as<intptr_t>();
      dim = apply_single_index(dim, reduction_ndim, NULL);
      reduction_dimflags[dim] = true;
    }
    else if (int_kind_array_tp.match(kw->axis.get_type())) {
      // An array of integer axes
      nd::with_1d_stride<intptr_t>(
          kw->axis, [&](intptr_t size, intptr_t stride, const intptr_t *data) {
            for (intptr_t i = 0; i < reduction_ndim; ++i) {
              reduction_dimflags[i] = false;
            }
            for (intptr_t i = 0; i < size; ++i) {
              intptr_t dim = data[i * stride];
              dim = apply_single_index(dim, reduction_ndim, NULL);
              reduction_dimflags[dim] = true;
            }
          });
    }
    else if (bool_kind_array_tp.match(kw->axis.get_type())) {
      // An array of boolean axes
      nd::with_1d_stride<dynd_bool>(
          kw->axis, [&](intptr_t size, intptr_t stride, const dynd_bool *data) {
            if (size > reduction_ndim) {
              stringstream ss;
              ss << "reduce: provided " << size
                 << " boolean values to axis parameter, maximum of "
                 << reduction_ndim;
              throw invalid_argument(ss.str());
            }
            // Copy the values provided, set the rest to false
            for (intptr_t i = 0; i < size; ++i) {
              reduction_dimflags[i] = data[i * stride];
            }
            for (intptr_t i = size; i < reduction_ndim; ++i) {
              reduction_dimflags[i] = false;
            }
          });
    }
    else {
      stringstream ss;
      ss << "dynd reduce: value " << kw->axis;
      ss << " is not valid as `axis` argument";
      throw invalid_argument(ss.str());
    }
  }

  // Count the number of dimensions being reduced
  intptr_t reducedim_count = 0;
  for (intptr_t i = 0; i < reduction_ndim; ++i) {
    reducedim_count += reduction_dimflags[i];
  }

  if (reducedim_count == 0) {
    if (reduction_ndim == 0) {
      // If there are no dimensions to reduce, it's
      // just a dst_initialization operation, so create
      // that ckernel directly
      if (!kw->dst_init.is_null()) {
        return kw->dst_init.get()->instantiate(
            kw->dst_init.get(), kw->dst_init.get_type(), NULL, ckb, ckb_offset,
            dst_tp, dst_arrmeta, kw->red_op.get_type()->get_npos(), src_tp,
            src_arrmeta, kernreq, ectx, nd::array(),
            std::map<nd::string, ndt::type>());
      }
      else if (kw->red_ident.is_null()) {
        return make_assignment_kernel(NULL, NULL, ckb, ckb_offset, dst_tp,
                                      dst_arrmeta, src_tp, src_arrmeta, kernreq,
                                      ectx, nd::array());
      }
      else {
        // Create the kernel which copies the identity and then
        // does one reduction
        return strided_inner_reduction_kernel_extra::make(
            kw->red_op, kw->dst_init, ckb, ckb_offset, 0, 1, dst_tp,
            dst_arrmeta, *src_tp, *src_arrmeta, kw->right_associative,
            kw->red_ident, kernreq, ectx);
      }
    }
    throw runtime_error("make_lifted_reduction_ckernel: no dimensions were "
                        "flagged for reduction");
  }

  if (!(reducedim_count == 1 || (kw->associative && kw->commutative))) {
    throw runtime_error(
        "make_lifted_reduction_ckernel: for reducing along multiple dimensions,"
        " the reduction function must be both associative and commutative");
  }
  if (kw->right_associative) {
    throw runtime_error("make_lifted_reduction_ckernel: right_associative is "
                        "not yet supported");
  }

  ndt::type dst_el_tp = kw->red_op.get_return_type();
  ndt::type src_el_tp = kw->red_op.get_pos_type(0);

  // This is the number of dimensions being processed by the reduction
  if (reduction_ndim != src_tp[0].get_ndim() - src_el_tp.get_ndim()) {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: wrong number of reduction "
          "dimensions, requested " << reduction_ndim << ", but types have ";
    ss << (src_tp[0].get_ndim() - src_el_tp.get_ndim());
    ss << " lifting from " << src_el_tp << " to " << src_tp;
    throw runtime_error(ss.str());
  }
  // Determine whether reduced dimensions are being kept or not
  bool keep_dims;
  if (reduction_ndim == dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
    keep_dims = true;
  }
  else if (reduction_ndim - reducedim_count ==
           dst_tp.get_ndim() - dst_el_tp.get_ndim()) {
    keep_dims = false;
  }
  else {
    stringstream ss;
    ss << "make_lifted_reduction_ckernel: The number of dimensions flagged for "
          "reduction, " << reducedim_count
       << ", is not consistent with the destination type reducing " << dst_tp
       << " with element " << dst_el_tp;
    throw runtime_error(ss.str());
  }

  ndt::type dst_i_tp = dst_tp, src_i_tp = src_tp[0];

  const char *src0_arrmeta = src_arrmeta[0];
  for (intptr_t i = 0; i < reduction_ndim; ++i) {
    intptr_t dst_stride, dst_size, src_stride, src_dim_size;
    // Get the striding parameters for the source dimension
    if (!src_i_tp.get_as_strided(src0_arrmeta, &src_dim_size, &src_stride,
                                 &src_i_tp, &src0_arrmeta)) {
      stringstream ss;
      ss << "make_lifted_reduction_ckernel: type " << src_i_tp
         << " not supported as source";
      throw type_error(ss.str());
    }
    if (reduction_dimflags[i]) {
      // This dimension is being reduced
      if (src_dim_size == 0 && kw->red_ident.is_null()) {
        // If the size of the src is 0, a reduction identity is required to get
        // a value
        stringstream ss;
        ss << "cannot reduce a zero-sized dimension (axis ";
        ss << i << " of " << src_i_tp << ") because the operation";
        ss << " has no identity";
        throw invalid_argument(ss.str());
      }
      if (keep_dims) {
        // If the dimensions are being kept, the output should be a
        // a strided dimension of size one
        if (dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                    &dst_i_tp, &dst_arrmeta)) {
          if (dst_size != 1 || dst_stride != 0) {
            stringstream ss;
            ss << "make_lifted_reduction_ckernel: destination of a reduction "
                  "dimension ";
            ss << "must have size 1, not size" << dst_size << "/stride "
               << dst_stride;
            ss << " in type " << dst_i_tp;
            throw type_error(ss.str());
          }
        } else {
          stringstream ss;
          ss << "make_lifted_reduction_ckernel: type " << dst_i_tp;
          ss << " not supported the destination of a dimension being reduced";
          throw type_error(ss.str());
        }
      }
      if (i < reduction_ndim - 1) {
        // An initial dimension being reduced
        ckb_offset = strided_initial_reduction_kernel_extra::make(
            ckb, ckb_offset, src_stride, src_dim_size, kernreq);
        // The next request should be single, as that's the kind of
        // ckernel the 'first_call' should be in this case
        kernreq = kernel_request_single;
      } else {
        // The innermost dimension being reduced
        return strided_inner_reduction_kernel_extra::make(
            kw->red_op, kw->dst_init, ckb, ckb_offset, src_stride, src_dim_size,
            dst_i_tp, dst_arrmeta, src_i_tp, src0_arrmeta,
            kw->right_associative, kw->red_ident, kernreq, ectx);
      }
    } else {
      // This dimension is being broadcast, not reduced
      if (!dst_i_tp.get_as_strided(dst_arrmeta, &dst_size, &dst_stride,
                                   &dst_i_tp, &dst_arrmeta)) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: type " << dst_i_tp
           << " not supported as destination";
        throw type_error(ss.str());
      }
      if (dst_size != src_dim_size) {
        stringstream ss;
        ss << "make_lifted_reduction_ckernel: the dst dimension size "
           << dst_size;
        ss << " must equal the src dimension size " << src_dim_size
           << " for broadcast dimensions";
        throw runtime_error(ss.str());
      }
      if (i < reduction_ndim - 1) {
        // An initial dimension being broadcast
        ckb_offset = strided_initial_broadcast_kernel_extra::make(
            ckb, ckb_offset, dst_stride, src_stride, src_dim_size, kernreq);
        // The next request should be strided, as that's the kind of
        // ckernel the 'first_call' should be in this case
        kernreq = kernel_request_strided;
      } else {
        // The innermost dimension being broadcast
        return strided_inner_broadcast_kernel_extra::make(
            kw->red_op, kw->dst_init, ckb, ckb_offset, dst_stride, src_stride,
            src_dim_size, dst_i_tp, dst_arrmeta, src_i_tp, src0_arrmeta,
            kw->right_associative, kw->red_ident, kernreq, ectx);
      }
    }
  }

  throw runtime_error("make_lifted_reduction_ckernel: internal error, "
                      "should have returned in the loop");
}
