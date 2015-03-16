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

    static void incr(const char *arrmeta, const char **data, intptr_t *index)
    {
      ++index[0];
      *data +=
          reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta)->stride;
    }
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

    static void incr(const char *arrmeta, const char **data, intptr_t *index)
    {
      const size_stride_t *size_stride =
          reinterpret_cast<const size_stride_t *>(arrmeta);

      if (++index[0] != size_stride->dim_size) {
        *data += size_stride->stride;
      } else {
        index[0] = 0;
        *data -= (size_stride->dim_size - 1) * size_stride->stride;
        T::incr(arrmeta + sizeof(size_stride_t), data, index + 1);
      }
    }
  };

} // namespace dynd::detail

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
    const size_stride_t *size_stride =
        reinterpret_cast<const size_stride_t *>(m_arrmeta);
    return size_stride->dim_size;
  }

  intptr_t get_stride() const
  {
    const size_stride_t *size_stride =
        reinterpret_cast<const size_stride_t *>(m_arrmeta);
    return size_stride->stride;
  }

  void set_data(const char *data) { m_data = data; }

  bool is_valid(const intptr_t *i) const
  {
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

  class iterator {
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
        fixed_dim::incr(m_values.get_arrmeta(), &m_data, m_index);
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

} // namespace dynd