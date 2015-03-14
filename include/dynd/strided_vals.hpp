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

template <typename T>
class builtin {
  const char *m_data;

public:
  builtin(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta),
          const start_stop_t *DYND_UNUSED(start_stop))
      : m_data(NULL)
  {
  }

  void set_data(const char *data) { m_data = data; }

  const T &operator()(const char *data, const intptr_t *DYND_UNUSED(i)) const
  {
    return *reinterpret_cast<const T *>(data);
  }

  typedef T dtype;

  class iterator {
  public:
    iterator(const builtin &DYND_UNUSED(values)) {}

    void get_index(intptr_t *DYND_UNUSED(i)) const {}

    bool is_valid() const { return true; }

    void incr(const char *&) {}
  };
};

// wrap_if_scalar
template <typename T>
using wrap_if_builtin =
    typename std::conditional<std::is_same<T, int>::value, builtin<T>, T>::type;

template <typename T>
class fixed_dim : public wrap_if_builtin<T> {
  typedef wrap_if_builtin<T> parent_type;

  size_stride_t m_size_stride;
  const char *m_data;
  const intptr_t *m_start;
  const intptr_t *m_stop;

public:
  typedef typename parent_type::dtype dtype;


  fixed_dim(const ndt::type &tp, const char *arrmeta,
            const start_stop_t *start_stop)
      : parent_type(tp.extended<fixed_dim_type>()->get_element_type(),
                    arrmeta + sizeof(size_stride_t), start_stop + 1),
        m_data(NULL), m_start(&start_stop->start), m_stop(&start_stop->stop)
  {
    m_size_stride = *reinterpret_cast<const size_stride_t *>(arrmeta);
  }

  const char *data() const { return m_data; }

  intptr_t get_size() const { return m_size_stride.dim_size; }

  size_t get_stride() const { return m_size_stride.stride; }

  void set_data(const char *data) { m_data = data; }

  bool is_valid(intptr_t i) const { return (i >= *m_start) && (i < *m_stop); }

  dtype operator()(const char *data, const intptr_t *i) const
  {
    return parent_type::operator()(data + i[0] * m_size_stride.stride, i + 1);
  }

  dtype operator()(const intptr_t *i) const { return (*this)(m_data, i); }

  dtype operator()(const std::initializer_list<intptr_t> &i) const
  {
    return (*this)(i.begin());
  }


  class iterator : public parent_type::iterator {
    const fixed_dim &m_values;
    const char *m_data;
    intptr_t m_index;

  public:
    iterator(const fixed_dim &values, intptr_t offset = 0)
        : parent_type::iterator(values), m_values(values),
          m_data(values.data() + offset), m_index(0)
    {
    }

    void get_index(intptr_t *i) const
    {
      i[0] = m_index;
      parent_type::iterator::get_index(i + 1);
    }

    bool is_valid() const
    {
      return m_values.is_valid(m_index) && parent_type::iterator::is_valid();
    }

    iterator &operator++()
    {
      do {
        incr(m_data);
      } while (*this != m_values.end() && !is_valid());

      return *this;
    }

    iterator operator++(int)
    {
      iterator tmp(*this);
      operator++();
      return tmp;
    }

    const int &operator*() const
    {
      return *reinterpret_cast<const int *>(m_data);
    }

    bool operator==(const iterator &other) const
    {
      return m_data == other.m_data;
    }

    bool operator!=(const iterator &other) const { return !(*this == other); }

    void incr(const char *&data)
    {
      if (++m_index != m_values.get_size()) {
        data += m_values.get_stride();
      } else if (!std::is_same<T, int>::value) {
        m_index = 0;
        data -= (m_values.get_size() - 1) * m_values.get_stride();
        parent_type::iterator::incr(data);
      }
    }
  };

  iterator begin() const
  {
    iterator it(*this);
    if (it.is_valid()) {
      return it;
    }

    return ++it;
  }

  iterator end() const
  {
    return iterator(*this, m_size_stride.dim_size * m_size_stride.stride);
  }
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