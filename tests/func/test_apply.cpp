//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class Apply;

typedef integral_constant<kernel_request_t, kernel_request_host>
    KernelRequestHost;

template <>
class Apply<KernelRequestHost> : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = KernelRequestHost::value;

  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a);
  }

  static nd::array To(nd::array a) { return a; }
};

#ifdef DYND_CUDA

typedef integral_constant<kernel_request_t, kernel_request_cuda_device>
    KernelRequestCUDADevice;

template <>
class Apply<KernelRequestCUDADevice> : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = KernelRequestCUDADevice::value;

  static nd::array To(const nd::array &a) { return a.to_cuda_device(); }

  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a).to_cuda_device();
  }
};

#endif

TYPED_TEST_CASE_P(Apply);

#define GET_HOST_FUNC(NAME)                                                    \
  template <kernel_request_t kernreq>                                          \
  typename std::enable_if<kernreq == kernel_request_host,                      \
                          decltype(&NAME)>::type get_##NAME()                  \
  {                                                                            \
    return &NAME;                                                              \
  }

int func0(int x, int y) { return 2 * (x - y); }

GET_HOST_FUNC(func0)

TEST(Apply, Function)
{
  typedef Apply<KernelRequestHost> TestFixture;

  nd::arrfunc af;

  af = nd::make_apply_arrfunc<kernel_request_host, decltype(&func0), &func0>();
  EXPECT_ARR_EQ(TestFixture::To(4), af(TestFixture::To(5), TestFixture::To(3)));
}

TEST(Apply, FunctionWithKeywords)
{
  typedef Apply<KernelRequestHost> TestFixture;

  nd::arrfunc af;

  af = nd::make_apply_arrfunc<decltype(&func0), &func0>("y");
  EXPECT_ARR_EQ(TestFixture::To(4),
                af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

  af = nd::make_apply_arrfunc<decltype(&func0), &func0>("x", "y");
  EXPECT_ARR_EQ(TestFixture::To(4),
                af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));
}

TYPED_TEST_P(Apply, Callable)
{
  nd::arrfunc af;

  if (TestFixture::KernelRequest == kernel_request_host) {
    af = nd::make_apply_arrfunc<kernel_request_host>(
        get_func0<kernel_request_host>());
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));
  }
}

TYPED_TEST_P(Apply, CallableWithKeywords)
{
  nd::arrfunc af;

  if (TestFixture::KernelRequest == kernel_request_host) {
    af = nd::make_apply_arrfunc<kernel_request_host>(
        get_func0<kernel_request_host>(), "y");
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), kwds("y", TestFixture::To(3))));

    af = nd::make_apply_arrfunc<kernel_request_host>(
        get_func0<kernel_request_host>(), "x", "y");
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));
  }
}

REGISTER_TYPED_TEST_CASE_P(Apply, Callable, CallableWithKeywords);

INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Apply, KernelRequestHost);

#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Apply, KernelRequestCUDADevice);
#endif