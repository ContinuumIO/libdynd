//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <mkl_dfti.h>

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace mkl {

    struct fft_kernel : base_strided_kernel<fft_kernel, 1> {
      DFTI_DESCRIPTOR_HANDLE descriptor;

      fft_kernel(size_t ndim, const char *src0_metadata) {
        MKL_LONG status;

        if (ndim == 1) {
          status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1,
                                        reinterpret_cast<const size_stride_t *>(src0_metadata)->dim_size);
        } else {
          MKL_LONG src0_size[3];
          for (size_t i = 0; i < ndim; ++i) {
            src0_size[i] = reinterpret_cast<const size_stride_t *>(src0_metadata)->dim_size;
            src0_metadata += sizeof(size_stride_t);
          }

          status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, ndim, src0_size);
        }

        status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

        status = DftiCommitDescriptor(descriptor);
      }

      ~fft_kernel() { DftiFreeDescriptor(&descriptor); }

      void single(char *dst, char *const *src) { DftiComputeForward(descriptor, src[0], dst); }
    };

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
