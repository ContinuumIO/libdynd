//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <iostream> // FOR DEBUG
#include <stdexcept>
#include <string>

#include <dynd/buffer.hpp>
#include <dynd/irange.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/var_dim_type.hpp>

namespace dynd {
namespace nd {

  /**
   * Constructs an uninitialized array of the given dtype. This is
   * the usual function to use for allocating such an array.
   */
  inline array empty(const ndt::type &tp);
  inline array empty(const ndt::type &tp, uint64_t flags);

  /** Stream printing function */
  DYND_API std::ostream &operator<<(std::ostream &o, const array &rhs);

  class array_vals;
  class array_vals_at;

  template <typename ValueType>
  struct as {
    void single(ValueType &value, char *data) const { value = *reinterpret_cast<ValueType *>(data); }
  };

  template <>
  struct as<std::string> {
    void single(std::string &value, char *data) const {
      value.assign(reinterpret_cast<string *>(data)->data(), reinterpret_cast<string *>(data)->size());
    }
  };

  /**
   * This is the primary multi-dimensional array class.
   */
  class DYND_API array : public buffer {
  public:
    using buffer::buffer;

    array() = default;

    array(const array &values, const ndt::type &tp)
        : buffer(empty_buffer((values.get_type().get_ndim() == tp.get_ndim())
                                  ? tp
                                  : values.get_type().with_replaced_dtype(tp, tp.get_ndim()))) {
      assign(values);
    }

    /**
     * Accesses a dynamic property of the array.
     *
     * \param name  The property to access.
     */
    array p(const char *name) const;

    array p(const std::string &name) const;

    /**
     * Converts a dynamic type property to an array.
     */
    static nd::array from_type_property(const std::pair<ndt::type, const char *> &pair);

    /**
     * Calls the dynamic function.
     */
    template <typename... ArgTypes>
    array f(const char *name, ArgTypes &&... args) const;

    array &operator+=(const array &rhs);
    array &operator-=(const array &rhs);
    array &operator*=(const array &rhs);
    array &operator/=(const array &rhs);

    /**
     * A helper for assigning to the values in 'this'. Normal assignment to
     * an array variable has reference semantics, the reference gets
     * overwritten to point to the new array. The 'vals' function provides
     * syntactic sugar for the 'val_assign' function, allowing for more
     * natural looking assignments.
     *
     * Example:
     *      array a(ndt::make_type<float>());
     *      a.vals() = 100;
     */
    array_vals vals() const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0) const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0, const irange &i1) const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0, const irange &i1, const irange &i2) const;

    /**
     * A helper for assigning to the values indexed in an array.
     */
    array_vals_at vals_at(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const;

    /**
     * Evaluates the array, attempting to do the minimum work
     * required. If the array is not ane expression, simply
     * returns it as is, otherwise evaluates into a new copy.
     */
    array eval() const;

    /**
     * Evaluates the array node into a newly allocated strided array,
     * with the requested access flags.
     *
     * \param access_flags  The access flags for the result, default immutable.
     */
    array eval_copy(uint32_t access_flags = 0) const;

    /**
     * Returns a view of the array as bytes (for POD) or the storage type,
     * peeling away any expression types or encodings.
     */
    array storage() const;

    /**
     * General irange-based indexing operation.
     *
     * \param nindices  The number of 'irange' indices.
     * \param indices  The array of indices to apply.
     * \param collapse_leading  If true, collapses the leading dimension
     *                          to simpler types where possible. If false,
     *                          does not. If you want to read values, typically
     *                          use true, if you want to write values, typically
     *                          use false.
     */
    array at_array(intptr_t nindices, const irange *indices, bool collapse_leading = true) const;

    /**
     * The function call operator is used for indexing. Overloading
     * operator[] isn't practical for multidimensional objects.
     */
    array operator()(const irange &i0) const { return at_array(1, &i0); }

    /** Indexing with two index values */
    array operator()(const irange &i0, const irange &i1) const {
      irange i[2] = {i0, i1};
      return at_array(2, i);
    }

    /** Indexing with three index values */
    array operator()(const irange &i0, const irange &i1, const irange &i2) const {
      irange i[3] = {i0, i1, i2};
      return at_array(3, i);
    }
    /** Indexing with four index values */
    array operator()(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const {
      irange i[4] = {i0, i1, i2, i3};
      return at_array(4, i);
    }
    /** Indexing with five index values */
    array operator()(const irange &i0, const irange &i1, const irange &i2, const irange &i3, const irange &i4) const {
      irange i[5] = {i0, i1, i2, i3, i4};
      return at_array(5, i);
    }

    array at(const irange &i0) const { return at_array(1, &i0); }

    /**
     * Assigns values from another array to this array.
     */
    array assign(const array &rhs, assign_error_mode error_mode = assign_error_fractional) const;

    /**
     * Assigns the "not available" value to this array.
     */
    array assign_na() const;

    /**
     * Casts the type of the array into the specified type.
     * This casts the entire type. If you want to cast the
     * array data type, use 'ucast' instead.
     *
     * \param tp  The type into which the array should be cast.
     */
    array cast(const ndt::type &tp) const;

    /**
     * Casts the array data type of the array into the specified type.
     *
     * \param uniform_dt  The type into which the array's
     *                    dtype should be cast.
     * \param replace_ndim  The number of array dimensions of
     *                       this type which should be replaced.
     *                       E.g. the value 1 could cast the last
     *                       array dimension and the array data type
     *                       to the replacement uniform_dt.
     */
    array ucast(const ndt::type &uniform_dt, intptr_t replace_ndim = 0) const;

    /**
     * Casts the array data type of the array into the type specified
     * as the template parameter.
     */
    template <class T>
    inline array ucast(intptr_t replace_ndim = 0) const {
      return ucast(ndt::make_type<T>(), replace_ndim);
    }

    /**
     * Attempts to view the data of the array as a new dynd type,
     * raising an error if it cannot be done.
     *
     * \param tp  The dynd type to view the entire data as.
     */
    array view(const ndt::type &tp) const;

    template <int N>
    inline array view(const char (&rhs)[N]) {
      return view(ndt::type(rhs));
    }

    template <typename T>
    typename std::enable_if<ndt::traits<T>::is_same_layout, T>::type view() const {
      return *reinterpret_cast<const T *>(cdata());
    }

    template <typename T>
    typename std::enable_if<!ndt::traits<T>::is_same_layout, T>::type view() {
      return T(get()->metadata(), const_cast<char *>(cdata()));
    }

    /**
     * Attempts to view the uniform type-level of the array as a
     * new dynd type, raising an error if it cannot be done.
     *
     * \param uniform_dt  The dynd type to view the uniform data as.
     * \param replace_ndim  The number of array dimensions to swallow
     *                       into the viewed uniform_dt.
     */
    array uview(const ndt::type &uniform_dt, intptr_t replace_ndim) const;

    /**
     * Permutes the dimensions of the array, returning a view to the result.
     * Only strided dimensions can be permuted and no dimension can be permuted
     * across a variable dimension. At present, there is no error checking.
     *
     * \param ndim The number of dimensions to permute. If `ndim' is less than
     *that
     *             of this array, the other dimensions will not be modified.
     * \param axes The permutation. It must be of length `ndim' and not contain
     *             a value greater or equal to `ndim'.
     */
    array permute(intptr_t ndim, const intptr_t *axes) const;

    /**
     * Rotates the dimensions of the array so the axis `from' becomes the axis
     * `to'.
     * At present, there cannot be any variable dimensions.
     */
    array rotate(intptr_t to, intptr_t from = 0) const;

    array rotate(intptr_t from = 0) const { return rotate(get_ndim() - 1, from); }

    /**
     * Transposes (reverses) the dimensions of the array.
     * At present, there cannot be any variable dimensions.
     */
    array transpose() const;

    /**
     * DEPRECATED
     * Views the array's memory as another type, where such an operation
     * makes sense. This is analogous to reinterpret_cast<>.
     */
    array view_scalars(const ndt::type &scalar_tp) const;

    /**
     * Replaces the array data type with a new one, returning a view to
     * the result. The new type must have the same storage as the
     * existing type.
     *
     * \param replacement_tp  The replacement type.
     * \param replace_ndim  The number of array dimensions to replace
     *                       in addition to the array data type.
     */
    array replace_dtype(const ndt::type &replacement_tp, intptr_t replace_ndim = 0) const;

    /**
     * Inserts new fixed dimensions into the array.
     *
     * \param i  The axis at which the new fixed dimensions start.
     * \param new_ndim  The number of array dimensions to insert.
     */
    array new_axis(intptr_t i, intptr_t new_ndim = 1) const;

    /**
     * Views the array's memory as another type, where such an operation
     * makes sense. This is analogous to reinterpret_case<>.
     */
    template <class T>
    array view_scalars() const {
      return view_scalars(ndt::make_type<T>());
    }

    /**
     * When this is a zero-dimensional array, converts it to a C++ scalar of the
     * requested template type. This function may be extended in the future for
     * 1D vectors (as<std::vector<T>>), matrices, etc.
     */
    template <typename ValueType>
    ValueType as(assign_error_mode error_mode = assign_error_fractional) const {
      ValueType value;
      nd::as<ValueType> as;

      ndt::type tp = ndt::make_type<ValueType>();
      if (tp == get_type()) {
        as.single(value, const_cast<char *>(cdata()));
      } else {
        array a = empty(tp);
        a.assign(*this, error_mode);

        as.single(value, const_cast<char *>(a.cdata()));
      }

      return value;
    }

    /** Returns a copy of this array in default memory. */
    array to_host() const;

#ifdef DYND_CUDA
    /** Returns a copy of this array in CUDA host memory. */
    array to_cuda_host(unsigned int cuda_host_flags = cudaHostAllocDefault) const;

    /** Returns a copy of this array in CUDA global memory. */
    array to_cuda_device() const;
#endif // DYND_CUDA

    bool is_na() const;

    /** Sorting comparison between two arrays. (Returns a bool, does not
     * broadcast) */
    //   bool op_sorting_less(const array &rhs) const;

    bool equals_exact(const array &rhs) const;

    friend DYND_API std::ostream &operator<<(std::ostream &o, const array &rhs);
    friend class array_vals;
    friend class array_vals_at;
  };

  DYND_API array tuple(size_t size, const array *vals);

  inline array tuple(std::initializer_list<array> vals) { return tuple(vals.size(), vals.begin()); }

  DYND_API array as_struct(size_t size, const std::pair<const char *, array> *pairs);

  inline array as_struct(const std::initializer_list<std::pair<const char *, array>> &pairs) {
    return as_struct(pairs.size(), pairs.begin());
  }

  DYND_API array operator+(const array &a0);
  DYND_API array operator-(const array &a0);
  DYND_API array operator!(const array &a0);
  DYND_API array operator~(const array &a0);

  DYND_API array operator+(const array &op0, const array &op1);
  DYND_API array operator-(const array &op0, const array &op1);
  DYND_API array operator/(const array &op0, const array &op1);
  DYND_API array operator*(const array &op0, const array &op1);
  DYND_API array operator%(const array &op0, const array &op1);

  DYND_API array operator&(const array &op0, const array &op1);
  DYND_API array operator|(const array &op0, const array &op1);
  DYND_API array operator^(const array &op0, const array &op1);
  DYND_API array operator<<(const array &op0, const array &op1);
  DYND_API array operator>>(const array &op0, const array &op1);

  DYND_API array operator&&(const array &a0, const array &a1);
  DYND_API array operator||(const array &a0, const array &a1);

  DYND_API array operator<(const array &a0, const array &a1);
  DYND_API array operator<=(const array &a0, const array &a1);
  DYND_API array operator==(const array &a0, const array &a1);
  DYND_API array operator!=(const array &a0, const array &a1);
  DYND_API array operator>=(const array &a0, const array &a1);
  DYND_API array operator>(const array &a0, const array &a1);

  /**
   * This is a helper class for dealing with value assignment and collapsing
   * a view-based array into a strided array. Only the array class itself
   * is permitted to construct this helper object, and it is non-copyable.
   *
   * All that can be done is assigning the values of the referenced array
   * to another array, or assigning values from another array into the elements
   * the referenced array.
   */
  class DYND_API array_vals {
    const array &m_arr;
    array_vals(const array &arr) : m_arr(arr) {}

    // Non-copyable, not default-constructable
    array_vals(const array_vals &);
    array_vals &operator=(const array_vals &);

  public:
    /**
     * Assigns values from an array to the internally referenced array.
     * this does a val_assign with the default assignment error mode.
     */
    array_vals &operator=(const array &rhs) {
      m_arr.assign(rhs);
      return *this;
    }

    /** Does a value-assignment from the rhs C++ scalar. */
    template <class T>
    typename std::enable_if<is_dynd_scalar<T>::value, array_vals &>::type operator=(const T &rhs) {
      m_arr.assign(rhs);
      return *this;
    }

    // TODO: Could also do +=, -=, *=, etc.

    friend class array;
    friend array_vals array::vals() const;
  };

  /**
   * This is a helper class like array_vals, but it holds a reference
   * to the temporary array. This is needed by vals_at.
   *
   */
  class DYND_API array_vals_at {
    array m_arr;
    array_vals_at(const array &arr) : m_arr(arr) {}

    array_vals_at(array &&arr) : m_arr(std::move(arr)) {}

    // Non-copyable, not default-constructable
    array_vals_at(const array_vals &);
    array_vals_at &operator=(const array_vals_at &);

  public:
    /**
     * Assigns values from an array to the internally referenced array.
     * this does a val_assign with the default assignment error mode.
     */
    array_vals_at &operator=(const array &rhs) {
      m_arr.assign(rhs);
      return *this;
    }

    /** Does a value-assignment from the rhs C++ scalar. */
    template <class T>
    typename std::enable_if<is_dynd_scalar<T>::value, array_vals_at &>::type operator=(const T &rhs) {
      m_arr.assign(rhs);
      return *this;
    }

    friend class array;
    friend array_vals_at array::vals_at(const irange &) const;
    friend array_vals_at array::vals_at(const irange &, const irange &) const;
    friend array_vals_at array::vals_at(const irange &, const irange &, const irange &) const;
    friend array_vals_at array::vals_at(const irange &, const irange &, const irange &, const irange &) const;
  };

  /**
   * \brief Makes a strided array pointing to existing data
   *
   * \param uniform_dtype  The type of each element in the strided array.
   * \param ndim  The number of strided dimensions.
   * \param shape  The shape of the strided dimensions.
   * \param strides  The strides of the strided dimensions.
   * \param access_flags Read/write/immutable flags.
   * \param data_ptr  Pointer to the element at index 0.
   * \param data_reference  A memory block which holds a reference to the data.
   * \param out_uniform_arrmeta  If the uniform_dtype has arrmeta
   *(get_arrmeta_size() > 0),
   *                              this must be non-NULL, and is populated with a
   *pointer to the
   *                              arrmeta for the uniform_dtype. The caller must
   *populate it
   *                              with valid data.
   *
   * \returns  The created array.
   */
  DYND_API array make_strided_array_from_data(const ndt::type &uniform_dtype, intptr_t ndim, const intptr_t *shape,
                                              const intptr_t *strides, int64_t access_flags, char *data_ptr,
                                              const memory_block &data_reference, char **out_uniform_arrmeta = NULL);

  inline array_vals array::vals() const { return array_vals(*this); }

  inline array_vals_at array::vals_at(const irange &i0) const { return array_vals_at(at_array(1, &i0, false)); }

  inline array_vals_at array::vals_at(const irange &i0, const irange &i1) const {
    irange i[2] = {i0, i1};
    return array_vals_at(at_array(2, i, false));
  }

  inline array_vals_at array::vals_at(const irange &i0, const irange &i1, const irange &i2) const {
    irange i[3] = {i0, i1, i2};
    return array_vals_at(at_array(3, i, false));
  }

  inline array_vals_at array::vals_at(const irange &i0, const irange &i1, const irange &i2, const irange &i3) const {
    irange i[4] = {i0, i1, i2, i3};
    return array_vals_at(at_array(4, i, false));
  }

  /**
   * Constructs an uninitialized array of the given dtype, with ndim/shape
   * pointer. This function is not named ``empty`` because (intptr_t, intptr_t,
   * type) and (intptr_t, const intptr_t *, type) can sometimes result in
   * unexpected overloads.
   */
  inline array dtyped_empty(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
    if (ndim > 0) {
      intptr_t i = ndim - 1;
      ndt::type rtp = shape[i] >= 0 ? ndt::make_fixed_dim(shape[i], tp) : ndt::make_type<ndt::var_dim_type>(tp);
      while (i-- > 0) {
        rtp = shape[i] >= 0 ? ndt::make_fixed_dim(shape[i], rtp) : ndt::make_type<ndt::var_dim_type>(rtp);
      }
      return empty(rtp);
    } else {
      return empty(tp);
    }
  }

  /**
   * A version of dtyped_empty that accepts a std::vector as the shape.
   */
  inline array dtyped_empty(const std::vector<intptr_t> &shape, const ndt::type &tp) {
    return dtyped_empty(shape.size(), shape.empty() ? NULL : &shape[0], tp);
  }

  /**
   * Constructs an uninitialized array of the given dtype,
   * specified as a string literal. This is a shortcut for expressions
   * like
   *
   * nd::array a = nd::empty("10 * int32");
   */
  template <int N>
  inline array empty(const char (&dshape)[N]) {
    return nd::empty(ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs a writable uninitialized array of the specified shape
   * and dtype. Prefixes the dtype with ``strided`` or ``var`` dimensions.
   */
  inline array empty(intptr_t dim0, const ndt::type &tp) {
    return nd::empty(dim0 >= 0 ? ndt::make_fixed_dim(dim0, tp) : ndt::make_type<ndt::var_dim_type>(tp));
  }

  /**
   * Constructs an uninitialized array of the given type,
   * specified as a string. This is a shortcut for expressions
   * like
   *
   *      array a = nd::empty(10, "int32");
   */
  template <int N>
  inline array empty(intptr_t dim0, const char (&dshape)[N]) {
    return empty(dim0, ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs a writable uninitialized array of the specified shape
   * and dtype. Prefixes the dtype with ``strided`` or ``var`` dimensions.
   */
  inline array empty(intptr_t dim0, intptr_t dim1, const ndt::type &tp) {
    ndt::type rtp = (dim1 >= 0) ? ndt::make_fixed_dim(dim1, tp) : ndt::make_type<ndt::var_dim_type>(tp);
    rtp = (dim0 >= 0) ? ndt::make_fixed_dim(dim0, rtp) : ndt::make_type<ndt::var_dim_type>(rtp);
    return nd::empty(rtp);
  }

  /**
   * Constructs an uninitialized array of the given type,
   * specified as a string. This is a shortcut for expressions
   * like
   *
   *      array a = nd::empty(10, 10, "int32");
   */
  template <int N>
  inline array empty(intptr_t dim0, intptr_t dim1, const char (&dshape)[N]) {
    return empty(dim0, dim1, ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs a writable uninitialized array of the specified type.
   * This type should be at least three dimensional, and is initialized
   * using the specified dimension sizes.
   */
  inline array empty(intptr_t dim0, intptr_t dim1, intptr_t dim2, const ndt::type &tp) {
    ndt::type rtp = (dim2 >= 0) ? ndt::make_fixed_dim(dim2, tp) : ndt::make_type<ndt::var_dim_type>(tp);
    rtp = (dim1 >= 0) ? ndt::make_fixed_dim(dim1, rtp) : ndt::make_type<ndt::var_dim_type>(rtp);
    rtp = (dim0 >= 0) ? ndt::make_fixed_dim(dim0, rtp) : ndt::make_type<ndt::var_dim_type>(rtp);
    return empty(rtp);
  }

  /**
   * Constructs an uninitialized array of the given type,
   * specified as a string. This is a shortcut for expressions
   * like
   *
   *      array a = nd::empty(10, 10, 10, "int32");
   */
  template <int N>
  inline array empty(intptr_t dim0, intptr_t dim1, intptr_t dim2, const char (&dshape)[N]) {
    return empty(dim0, dim1, dim2, ndt::type(dshape, dshape + N - 1));
  }

  /**
   * Constructs an array with the same shape and memory layout
   * of the one given, but replacing the
   *
   * \param rhs  The array whose shape and memory layout to emulate.
   * \param uniform_dtype   The array data type of the new array.
   */
  DYND_API array empty_like(const array &rhs, const ndt::type &uniform_dtype);

  /**
   * Constructs an empty array matching the parameters of 'rhs'
   *
   * \param rhs  The array whose shape, memory layout, and dtype to emulate.
   */
  DYND_API array empty_like(const array &rhs);

  /**
   * Constructs an array, with each element initialized to 0, of the given
   * dtype,
   * with ndim/shape pointer.
   */
  inline array dtyped_zeros(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
    nd::array res = dtyped_empty(ndim, shape, tp);
    res.assign(0);

    return res;
  }

  inline array zeros(intptr_t dim0, const ndt::type &tp) {
    intptr_t shape[1] = {dim0};

    return dtyped_zeros(1, shape, tp);
  }
  inline array zeros(intptr_t dim0, intptr_t dim1, const ndt::type &tp) {
    intptr_t shape[2] = {dim0, dim1};

    return dtyped_zeros(2, shape, tp);
  }

  /**
   * Primitive function to construct an nd::array with each element initialized
   * to 1.
   * In this function, the type provided is the complete type of the array
   * result, not just its dtype.
   */
  DYND_API array typed_ones(intptr_t ndim, const intptr_t *shape, const ndt::type &tp);

  /**
   * A version of typed_ones that accepts a std::vector as the shape.
   */
  inline array ones(const std::vector<intptr_t> &shape, const ndt::type &tp) {
    return typed_ones(shape.size(), shape.empty() ? NULL : &shape[0], tp);
  }

  /**
   * Constructs an array, with each element initialized to 1, of the given
   * dtype,
   * with ndim/shape pointer.
   */
  inline array dtyped_ones(intptr_t ndim, const intptr_t *shape, const ndt::type &tp) {
    nd::array res = dtyped_empty(ndim, shape, tp);
    res.assign(1);

    return res;
  }

  inline array ones(intptr_t dim0, const ndt::type &tp) {
    intptr_t shape[1] = {dim0};

    return dtyped_ones(1, shape, tp);
  }
  inline array ones(intptr_t dim0, intptr_t dim1, const ndt::type &tp) {
    intptr_t shape[2] = {dim0, dim1};

    return dtyped_ones(2, shape, tp);
  }

  /**
   * This concatenates two 1D arrays. It is really just a placeholder until a
   * proper
   * concatenate is written. It shouldn't be used unless you absolutely know
   * what you
   * are doing. It needs to be implemented properly.
   */
  DYND_API array concatenate(const nd::array &x, const nd::array &y);

  /**
   * Reshapes an array into the new shape. This is currently a prototype and
   * should only be used
   * with contiguous arrays of built-in dtypes.
   */
  DYND_API array reshape(const array &a, const array &shape);

  /**
   * Reshapes an array into the new shape. This is currently a prototype and
   * should only be used
   * with contiguous arrays of built-in dtypes.
   */
  inline array reshape(const array &a, intptr_t ndim, const intptr_t *shape) {
    return reshape(a, nd::array(shape, ndim));
  }

  /**
   * Memory-maps a file with dynd type 'bytes'.
   *
   * \param filename  The name of the file to memory map.
   * \param begin  If provided, the start of where to memory map. Uses
   *               Python semantics for out of bounds and negative values.
   * \param end  If provided, the end of where to memory map. Uses
   *             Python semantics for out of bounds and negative values.
   * \param access  The access permissions with which to open the file.
   */
  DYND_API array memmap(const std::string &filename, intptr_t begin = 0,
                        intptr_t end = std::numeric_limits<intptr_t>::max(), uint32_t access = default_access_flags);

  /**
   * Creates a ctuple nd::array with the given field names and
   * pointers to the provided field values.
   *
   * \param  field_count  The number of fields.
   * \param  field_values  The values of the fields.
   */
  DYND_API array combine_into_tuple(size_t field_count, const array *field_values);

  /**
   * Creates a memory block for holding an nd::array (i.e. a container for nd::array arrmeta)
   *
   * The created object is uninitialized.
   */
  inline array make_array(const ndt::type &tp, uint64_t flags) {
    if (tp.is_symbolic()) {
      std::stringstream ss;
      ss << "Cannot create a dynd array with symbolic type " << tp;
      throw type_error(ss.str());
    }

    size_t data_offset = inc_to_alignment(sizeof(buffer_memory_block) + tp.get_arrmeta_size(), tp.get_data_alignment());
    size_t data_size = tp.get_default_data_size();

    return array(new (data_offset + data_size - sizeof(buffer_memory_block))
                     buffer_memory_block(tp, data_offset, data_size, flags),
                 false);
  }

  inline array make_array(const ndt::type &tp, char *data, uint64_t flags) {
    return array(new (tp.get_arrmeta_size()) buffer_memory_block(tp, data, flags), false);
  }

  inline array make_array(const ndt::type &tp, char *data, const memory_block &owner, uint64_t flags) {
    return array(new (tp.get_arrmeta_size()) buffer_memory_block(tp, data, owner, flags), false);
  }

  inline array empty(const ndt::type &tp, uint64_t flags) {
    // Create an empty shell
    array res = make_array(tp, flags);
    // Construct the arrmeta with default settings
    if (tp.get_arrmeta_size() > 0) {
      res.get_type()->arrmeta_default_construct(res->metadata(), true);
    }

    return res;
  }

  inline array empty(const ndt::type &tp) {
    // (tp.get_ndim() == 0) ? (read_access_flag | immutable_access_flag) : readwrite_access_flags
    return empty(tp, readwrite_access_flags);
  }

  /**
   * Makes a shallow copy of the nd::array memory block. In the copy, only the
   * nd::array arrmeta is duplicated, all the references are the same. Any NULL
   * references are swapped to point at the original nd::array memory block, as they
   * are a signal that the data was embedded in the same memory allocation.
   */
  DYND_API array shallow_copy_array_memory_block(const array &ndo);

} // namespace dynd::nd

namespace ndt {

  /**
   * Does a value lookup into an array of type "N * T", without
   * bounds checking the index ``i`` or validating that ``a`` has the
   * required type. Use only when these checks have been done externally.
   */
  template <typename T>
  inline const T &unchecked_fixed_dim_get(const nd::array &a, intptr_t i) {
    const size_stride_t *md = reinterpret_cast<const size_stride_t *>(a.get()->metadata());
    return *reinterpret_cast<const T *>(a.cdata() + i * md->stride);
  }

  /**
   * Does a writable value lookup into an array of type "N * T", without
   * bounds checking the index ``i`` or validating that ``a`` has the
   * required type. Use only when these checks have been done externally.
   */
  template <typename T>
  inline T &unchecked_fixed_dim_get_rw(const nd::array &a, intptr_t i) {
    const size_stride_t *md = reinterpret_cast<const size_stride_t *>(a.get()->metadata());
    return *reinterpret_cast<T *>(a.data() + i * md->stride);
  }

} // namespace dynd::ndt

/**
 * This function broadcasts the input array's shapes together,
 * producing a broadcast shape as the result. For any dimension in
 * an input with a variable-sized shape, the output shape is set
 * to a negative value.
 *
 * \param ninputs  The number of inputs whose shapes are to be broadcasted.
 * \param inputs  The inputs whose shapes are to be broadcasted.
 * \param out_undim  The number of dimensions in the output shape.
 * \param out_shape  This is filled with the broadcast shape.
 * \param out_axis_perm  A permutation of the axis for the output to use to
 *                       match the input's memory ordering.
 */
DYND_API void broadcast_input_shapes(intptr_t ninputs, const nd::array *inputs, intptr_t &out_undim,
                                     dimvector &out_shape, shortvector<int> &out_axis_perm);

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 *
 * \param ndim        The number of dimensions being broadcast to.
 * \param shape       The shape being broadcast to.
 * \param src_ndim    The number of dimensions of the input which is to be broadcast.
 * \param src_shape   The shape of the input which is to be broadcast.
 * \param src_strides The strides of the input which is to be broadcast.
 * \param out_strides The resulting strides after broadcasting (with length 'ndim').
 */
DYND_API void broadcast_to_shape(intptr_t ndim, const intptr_t *shape, intptr_t src_ndim, const intptr_t *src_shape,
                                 const intptr_t *src_strides, intptr_t *out_strides);

/**
 * Adjusts out_shape to broadcast it with the input shape.
 *
 * \param out_undim  The number of dimensions in the output
 *                   broadcast shape. This should be set to
 *                   the maximum of all the input undim values
 *                   that will be incrementally broadcasted.
 * \param out_shape  The shape that gets updated to become the
 *                   final broadcast shape. This should be
 *                   initialized to all ones before incrementally
 *                   broadcasting.
 * \param undim  The number of dimensions in the input shape.
 * \param shape  The input shape.
 */
DYND_API void incremental_broadcast(intptr_t out_undim, intptr_t *out_shape, intptr_t undim, const intptr_t *shape);

} // namespace dynd
