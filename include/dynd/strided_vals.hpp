//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

namespace dynd {

struct start_stop_t {
  intptr_t start;
  intptr_t stop;
};

namespace detail {

  template <int N>
  struct strided_utils {
    static const char *get(const char *pointer, const intptr_t *index,
                           const intptr_t *strides)
    {
      return strided_utils<N - 1>::get(pointer, strides, index) +
             index[N - 1] * strides[N - 1];
    }

    static bool is_valid(const intptr_t *index, const start_stop_t *start_stop)
    {
      return strided_utils<N - 1>::is_valid(index, start_stop) &&
             (index[N - 1] >= start_stop[N - 1].start) &&
             (index[N - 1] < start_stop[N - 1].stop);
    }

    static void incr(const char *&pointer, intptr_t *index,
                     const intptr_t *sizes, const intptr_t *strides)
    {
      if (++index[N - 1] != sizes[N - 1]) {
        pointer += strides[N - 1];
      } else {
        index[N - 1] = 0;
        pointer -= (sizes[N - 1] - 1) * strides[N - 1];
        strided_utils<N - 1>::incr(pointer, index, sizes, strides);
      }
    }
  };

  template <>
  struct strided_utils<1> {
    static const char *get(const char *pointer, const intptr_t *index,
                           const intptr_t *strides)
    {
      return pointer + index[0] * strides[0];
    }

    static bool is_valid(const intptr_t *index, const start_stop_t *start_stop)
    {
      return (index[0] >= start_stop[0].start) &&
             (index[0] < start_stop[0].stop);
    }

    static void incr(const char *&pointer, intptr_t *index,
                     const intptr_t *DYND_UNUSED(sizes),
                     const intptr_t *strides)
    {
      ++index[0];
      pointer += strides[0];
    }
  };

} // namespace detail

// return child.at(m_data + index[0] * m_stride, index + 1);

// fixed_dim_type
// fixed_dim_data
// fixed_dim_accessor

// fixed_dim
// fixed_dim_iterator

namespace detail {

  /*
    template <typename T>
    struct is_dim_type {
      static const bool value = false;
    };

    template <typename T>
    struct is_dim_type<fixed_dim<T>> {
      static const bool value = true;
    };

    template <typename T>
    class fixed_dim {
      const char *m_data;
      intptr_t m_size;
      intptr_t m_stride;
      const start_stop_t *m_start_stop;

    public:
      const char *get_data() const { return m_data; }

      intptr_t get_ndim() const { return 1; }

      intptr_t get_size(intptr_t DYND_UNUSED(i)) const { return m_size; }

      const intptr_t *get_sizes() const { return &m_size; }

      intptr_t get_stride(intptr_t DYND_UNUSED(i)) const { return m_stride; }

      const intptr_t *get_strides() const { return &m_stride; }

      void set_data(const char *data) { m_data = data; }

      void set_data(const char *data, const size_stride_t *size_stride,
                    const start_stop_t *start_stop = NULL)
      {
        m_data = data;
        for (intptr_t i = 0; i < 1; ++i) {
          m_stride = size_stride[i].stride;
          m_size = size_stride[i].dim_size;
        }
        m_start_stop = start_stop;
      }

      const int &operator()(const intptr_t *index) const
      {
        return *reinterpret_cast<const int *>(m_data + index[0] * m_stride);
      }

      const int &operator()(intptr_t i) const { return (*this)(&i); }

      bool is_valid(const intptr_t *index) const
      {
        return detail::strided_utils<1>::is_valid(index, m_start_stop);
      }

      class iterator {
        const fixed_dim<T> &m_vals;
        const char *m_data;
        intptr_t m_index[1];

      public:
        iterator(const fixed_dim<T> &vals, intptr_t offset = 0)
            : m_vals(vals), m_data(vals.get_data() + offset)
        {
          memset(m_index, 0, 1 * sizeof(intptr_t));
        }

        const intptr_t *get_index() const { return m_index; }

        iterator &operator++()
        {
          do {
            detail::strided_utils<1>::incr(m_data, m_index, m_vals.get_sizes(),
                                           m_vals.get_strides());
          } while (*this != m_vals.end() && !(m_vals.is_valid(m_index)));
          return *this;
        }

        iterator operator++(int)
        {
          iterator tmp(*this);
          operator++();
          return tmp;
        }

        bool operator==(const iterator &other) const
        {
          return &m_vals == &other.m_vals && m_data == other.m_data;
        }

        bool operator!=(const iterator &other) const { return !(*this == other);
    }

        const T &operator*() const
        {
          return *reinterpret_cast<const T *>(m_data);
        }
      };

      iterator begin() const
      {
        iterator it(*this);
        if (is_valid(it.get_index())) {
          return it;
        }

        return ++it;
      }

      iterator end() const { return iterator(*this, m_size * m_stride); }
    };
  */

} // namespace detail

namespace detail {

  template <typename T, bool inner>
  class fixed_dim;

  template <typename T>
  class fixed_dim<T, true> {
  public:
    static const size_t arrmeta_size = sizeof(fixed_dim_type_arrmeta);
    static const intptr_t ndim = 1;
    typedef T dtype;

    static int at(const intptr_t *i, const char *arrmeta, const char *data)
    {
      return *reinterpret_cast<const T *>(
                 data +
                 i[0] *
                     reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta)
                         ->stride);
    }

    class iterator {
    public:
      static void incr(const char *arrmeta, const char **data, intptr_t *index)
      {
        ++index[0];
        *data +=
            reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta)->stride;
      }
    };
  };

  template <typename T>
  class fixed_dim<T, false> {
  public:
    static const size_t arrmeta_size =
        sizeof(fixed_dim_type_arrmeta) + T::arrmeta_size;
    static const intptr_t ndim = 1 + T::ndim;
    typedef typename T::dtype dtype;

    static int at(const intptr_t *i, const char *arrmeta, const char *data)
    {
      return T::at(
          i + 1, arrmeta + sizeof(size_stride_t),
          data +
              i[0] * reinterpret_cast<const size_stride_t *>(arrmeta)->stride);
    }

    class iterator {
    public:
      static void incr(const char *arrmeta, const char **data, intptr_t *index)
      {
        const size_stride_t *size_stride =
            reinterpret_cast<const size_stride_t *>(arrmeta);

        if (++index[0] != size_stride->dim_size) {
          *data += size_stride->stride;
        } else {
          index[0] = 0;
          *data -= (size_stride->dim_size - 1) * size_stride->stride;
          T::iterator::incr(arrmeta + sizeof(size_stride_t), data, index + 1);
        }
      }
    };
  };

} // namespace dynd::detail

template <typename T>
class builtin {
public:
  static const intptr_t ndim = 0;
};

// wrap_if_scalar
template <typename T>
using wrap_if_builtin =
    typename std::conditional<std::is_same<T, int>::value, builtin<T>, T>::type;

template <typename T>
struct is_dim {
  static const bool value = false;
};

template <typename T>
struct is_dim<fixed_dim<T>> {
  static const bool value = true;
};

template <typename T>
class fixed_dim : public detail::fixed_dim<T, !is_dim<T>::value> {
  char m_arrmeta[fixed_dim::arrmeta_size];
  const char *m_data;
  const start_stop_t *m_start_stop;

public:
  typedef typename detail::fixed_dim<T, !is_dim<T>::value>::dtype dtype;

  fixed_dim(const ndt::type &tp, const char *arrmeta,
            const start_stop_t *start_stop = NULL)
      : m_data(NULL), m_start_stop(start_stop)
  {
    tp.extended()->arrmeta_copy_construct(m_arrmeta, arrmeta, NULL);
  }

  const char *data() const { return m_data; }

  const char *get_arrmeta() const { return m_arrmeta; }

  intptr_t get_size() const
  {
    return reinterpret_cast<const size_stride_t *>(m_arrmeta)->dim_size;
  }

  intptr_t get_stride() const
  {
    return reinterpret_cast<const fixed_dim_type_arrmeta *>(m_arrmeta)->stride;
  }

  void set_data(const char *data) { m_data = data; }

  bool is_valid(const intptr_t *i) const
  {
    if (m_start_stop == NULL) {
      return true;
    }

    for (intptr_t j = 0; j < fixed_dim::ndim; ++j) {
      if ((i[j] < m_start_stop[j].start) || (i[j] >= m_start_stop[j].stop)) {
        return false;
      }
    }

    return true;
  }

  const dtype &operator()(const intptr_t *i) const
  {
    return fixed_dim::at(i, m_arrmeta, m_data);
  }

  const dtype &operator()(const std::initializer_list<intptr_t> &i) const
  {
    return (*this)(i.begin());
  }

  class iterator : public detail::fixed_dim<T, !is_dim<T>::value>::iterator {
    const fixed_dim &m_values;
    const char *m_data;
    intptr_t m_index[fixed_dim::ndim];

  public:
    iterator(const fixed_dim &values, intptr_t offset = 0)
        : m_values(values), m_data(values.data() + offset)
    {
      memset(m_index, 0, sizeof(m_index));
    }

    const intptr_t *get_index() const { return m_index; }

    bool is_valid() const { return m_values.is_valid(m_index); }

    iterator &operator++()
    {
      do {
        iterator::incr(m_values.get_arrmeta(), &m_data, m_index);
      } while (*this != m_values.end() && !is_valid());

      return *this;
    }

    iterator operator++(int)
    {
      iterator tmp(*this);
      operator++();
      return tmp;
    }

    const dtype &operator*() const
    {
      return *reinterpret_cast<const dtype *>(m_data);
    }

    bool operator==(const iterator &other) const
    {
      return m_data == other.m_data;
    }

    bool operator!=(const iterator &other) const { return !(*this == other); }
  };

  iterator begin() const
  {
    iterator it(*this);
    if (it.is_valid()) {
      return it;
    }

    return ++it;
  }

  iterator end() const { return iterator(*this, get_size() * get_stride()); }
};

/*
template <typename T>
class strided_vals<T, 2> : public detail::strided_vals<T, 2> {
public:
    const T &operator ()(const intptr_t *index) const {
        return detail::strided_vals<T, 2>::operator ()(index);
    }

    const T &operator ()(intptr_t i0, intptr_t i1) const {
        const intptr_t index[2] = {i0, i1};
        return operator ()(index);
    }

    bool is_masked(const intptr_t *index) const {
        return detail::strided_vals<T, 2>::is_masked(index);
    }

    bool is_masked(intptr_t i0, intptr_t i1) const {
        const intptr_t index[2] = {i0, i1};
        return is_masked(index);
    }

    bool is_valid(const intptr_t *index) const {
        return detail::strided_vals<T, 2>::is_valid(index);
    }

    bool is_valid(intptr_t i0, intptr_t i1) const {
        const intptr_t index[2] = {i0, i1};
        return is_valid(index);
    }
};

template <typename T>
class strided_vals<T, 3> : public detail::strided_vals<T, 3> {
public:
    const T &operator ()(const intptr_t *index) const {
        return detail::strided_vals<T, 3>::operator ()(index);
    }

    const T &operator ()(intptr_t i0, intptr_t i1, intptr_t i2) const {
        const intptr_t index[3] = {i0, i1, i2};
        return operator ()(index);
    }

    bool is_masked(const intptr_t *index) const {
        return detail::strided_vals<T, 3>::is_masked(index);
    }

    bool is_masked(intptr_t i0, intptr_t i1, intptr_t i2) const {
        const intptr_t index[3] = {i0, i1, i2};
        return is_masked(index);
    }

    bool is_valid(const intptr_t *index) const {
        return detail::strided_vals<T, 3>::is_valid(index);
    }

    bool is_valid(intptr_t i0, intptr_t i1, intptr_t i2) const {
        const intptr_t index[3] = {i0, i1, i2};
        return is_valid(index);
    }
};
*/

} // namespace dynd