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

template <kernel_request_t kernreq, typename func_type, func_type func>
struct func_wrapper;

#if !(defined(_MSC_VER) && _MSC_VER == 1800)
#define FUNC_WRAPPER(KERNREQ, ...)                                             \
  template <typename R, typename... A, R (*func)(A...)>                        \
  struct func_wrapper<KERNREQ, R (*)(A...), func> {                            \
    __VA_ARGS__ R operator()(A... a) const { return (*func)(a...); }           \
  };
#else
// Workaround for MSVC 2013 variadic template bug
// https://connect.microsoft.com/VisualStudio/Feedback/Details/1034062
#define FUNC_WRAPPER(KERNREQ, ...)                                             \
  template <typename R, R (*func)()>                                           \
  struct func_wrapper<KERNREQ, R (*)(), func> {                                \
    __VA_ARGS__ R operator()() const { return (*func)(); }                     \
  };                                                                           \
  template <typename R, typename A0, R (*func)(A0)>                            \
  struct func_wrapper<KERNREQ, R (*)(A0), func> {                              \
    __VA_ARGS__ R operator()(A0 a0) const { return (*func)(a0); }              \
  };                                                                           \
  template <typename R, typename A0, typename A1, R (*func)(A0, A1)>           \
  struct func_wrapper<KERNREQ, R (*)(A0, A1), func> {                          \
    __VA_ARGS__ R operator()(A0 a0, A1 a1) const { return (*func)(a0, a1); }   \
  };                                                                           \
  template <typename R, typename A0, typename A1, typename A2,                 \
            R (*func)(A0, A1, A2)>                                             \
  struct func_wrapper<KERNREQ, R (*)(A0, A1, A2), func> {                      \
    __VA_ARGS__ R operator()(A0 a0, A1 a1, A2 a2) const                        \
    {                                                                          \
      return (*func)(a0, a1, a2);                                              \
    }                                                                          \
  }
#endif

FUNC_WRAPPER(kernel_request_host);

#undef FUNC_WRAPPER

int func0(int x, int y) { return 2 * (x - y); }

GET_HOST_FUNC(func0)

template <kernel_request_t kernreq>
struct func0_as_callable : func_wrapper<kernreq, decltype(&func0), &func0> {
};

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

    af = nd::make_apply_arrfunc(func0_as_callable<kernel_request_host>());
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(TestFixture::To(5), TestFixture::To(3)));

    af = nd::make_apply_arrfunc<kernel_request_host,
                                func0_as_callable<kernel_request_host>>();
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

    af = nd::make_apply_arrfunc(func0_as_callable<kernel_request_host>(), "y");
    EXPECT_ARR_EQ(TestFixture::To(4), af(5, kwds("y", TestFixture::To(3))));

    af = nd::make_apply_arrfunc(func0_as_callable<kernel_request_host>(), "x",
                                "y");
    EXPECT_ARR_EQ(TestFixture::To(4),
                  af(kwds("x", TestFixture::To(5), "y", TestFixture::To(3))));
  }
}

REGISTER_TYPED_TEST_CASE_P(Apply, Callable, CallableWithKeywords);

INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Apply, KernelRequestHost);

#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Apply, KernelRequestCUDADevice);
#endif