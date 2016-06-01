//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/diagnostics.hpp>

namespace dynd {

/**
 * Function for byteswapping a single value.
 */
inline uint16_t byteswap_value(uint16_t value) { return ((value & 0xffu) << 8) | (value >> 8); }

/**
 * Function for byteswapping a single value.
 */
inline uint32_t byteswap_value(uint32_t value)
{
  return ((value & 0xffu) << 24) | ((value & 0xff00u) << 8) | ((value & 0xff0000u) >> 8) | (value >> 24);
}

/**
 * Function for byteswapping a single value.
 */
inline uint64_t byteswap_value(uint64_t value)
{
  return ((value & 0xffULL) << 56) | ((value & 0xff00ULL) << 40) | ((value & 0xff0000ULL) << 24) |
         ((value & 0xff000000ULL) << 8) | ((value & 0xff00000000ULL) >> 8) | ((value & 0xff0000000000ULL) >> 24) |
         ((value & 0xff000000000000ULL) >> 40) | (value >> 56);
}

namespace nd {

  struct byteswap_ck : base_strided_kernel<byteswap_ck, 1> {
    size_t data_size;

    byteswap_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      // Do a different loop for in-place swap versus copying swap,
      // so this one kernel function works correctly for both cases.
      if (src[0] == dst) {
        // In-place swap
        for (size_t j = 0; j < data_size / 2; ++j) {
          std::swap(dst[j], dst[data_size - j - 1]);
        }
      }
      else {
        for (size_t j = 0; j < data_size; ++j) {
          dst[j] = src[0][data_size - j - 1];
        }
      }
    }
  };

  struct pairwise_byteswap_ck : base_strided_kernel<pairwise_byteswap_ck, 1> {
    size_t data_size;

    pairwise_byteswap_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      // Do a different loop for in-place swap versus copying swap,
      // so this one kernel function works correctly for both cases.
      if (src[0] == dst) {
        // In-place swap
        for (size_t j = 0; j < data_size / 4; ++j) {
          std::swap(dst[j], dst[data_size / 2 - j - 1]);
        }
        for (size_t j = 0; j < data_size / 4; ++j) {
          std::swap(dst[data_size / 2 + j], dst[data_size - j - 1]);
        }
      }
      else {
        for (size_t j = 0; j < data_size / 2; ++j) {
          dst[j] = src[0][data_size / 2 - j - 1];
        }
        for (size_t j = 0; j < data_size / 2; ++j) {
          dst[data_size / 2 + j] = src[0][data_size - j - 1];
        }
      }
    }
  };

  extern DYND_API callable byteswap;
  extern DYND_API callable pairwise_byteswap;

} // namespace dynd::nd
} // namespace dynd
