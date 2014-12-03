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

#ifdef __CUDACC__
#define CUDA_DEVICE_IF_CUDA_ELSE_HOST __device__
#define KERNREQ_CUDA_DEVICE_IF_CUDA_ELSE_KERNREQ_HOST kernel_request_cuda_device
#else
#define CUDA_DEVICE_IF_CUDA_ELSE_HOST
#define KERNREQ_CUDA_DEVICE_IF_CUDA_ELSE_KERNREQ_HOST kernel_request_host
#endif

using namespace std;
using namespace dynd;

template <typename T>
class Apply;

template <>
class Apply<integral_constant<kernel_request_t, kernel_request_host>>
    : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = kernel_request_host;

  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a);
  }

  static nd::array To(nd::array a) { return a; }
};

#ifdef DYND_CUDA

template <>
class Apply<integral_constant<kernel_request_t, kernel_request_cuda_device>>
    : public ::testing::Test {
public:
  static const kernel_request_t KernelRequest = kernel_request_cuda_device;

  static nd::array To(const nd::array &a) { return a.to_cuda_device(); }

  template <typename T>
  static nd::array To(const std::initializer_list<T> &a)
  {
    return nd::array(a).to_cuda_device();
  }
};

#endif

TYPED_TEST_CASE_P(Apply);

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

#ifdef __CUDACC__

FUNC_WRAPPER(kernel_request_cuda_device, __device__);

#endif

#undef FUNC_WRAPPER

#define GET_HOST_FUNC(NAME)                                                    \
  template <kernel_request_t kernreq>                                          \
  typename std::enable_if<kernreq == kernel_request_host,                      \
                          decltype(&NAME)>::type get_##NAME()                  \
  {                                                                            \
    return &NAME;                                                              \
  }

#define GET_CUDA_DEVICE_FUNC(NAME)                                             \
  __global__ void get_cuda_device_##NAME(void *res)                            \
  {                                                                            \
    *reinterpret_cast<decltype(&NAME) *>(res) = &NAME;                         \
  }                                                                            \
                                                                               \
  template <kernel_request_t kernreq>                                          \
  typename std::enable_if<kernreq == kernel_request_cuda_device,               \
                          decltype(&NAME)>::type get_##NAME()                  \
  {                                                                            \
    decltype(&NAME) res;                                                       \
    decltype(&NAME) *func, *cuda_device_func;                                  \
    throw_if_not_cuda_success(                                                 \
        cudaHostAlloc(&func, sizeof(decltype(&NAME)), cudaHostAllocMapped));   \
    throw_if_not_cuda_success(                                                 \
        cudaHostGetDevicePointer(&cuda_device_func, func, 0));                 \
    get_cuda_device_##NAME<<<1, 1>>>(cuda_device_func);                        \
    throw_if_not_cuda_success(cudaDeviceSynchronize());                        \
    res = *func;                                                               \
    throw_if_not_cuda_success(cudaFreeHost(func));                             \
                                                                               \
    return res;                                                                \
  }

#ifdef __CUDACC_
#define GET_CUDA_HOST_DEVICE_FUNC GET_HOST_FUNC
#else
#define GET_CUDA_HOST_DEVICE_FUNC(NAME)                                        \
  GET_HOST_FUNC(NAME)                                                          \
  GET_CUDA_DEVICE_FUNC(NAME)
#endif

#define FUNC_AS_CALLABLE(NAME)                                                 \
  template <kernel_request_t kernreq>                                          \
  struct NAME##_as_callable : func_wrapper<kernreq, decltype(&NAME), &NAME> {  \
  };

int func0(int x, int y) { return 2 * (x - y); }

GET_HOST_FUNC(func0)
FUNC_AS_CALLABLE(func0);

#ifdef __CUDACC__

__device__ double func1(double x, int y) { return x + 2.6 * y; }

GET_CUDA_DEVICE_FUNC(func1)
FUNC_AS_CALLABLE(func1);

#endif

DYND_CUDA_HOST_DEVICE float func2(const float (&x)[3])
{
  return x[0] + x[1] + x[2];
}

GET_CUDA_HOST_DEVICE_FUNC(func2)
FUNC_AS_CALLABLE(func2);

DYND_CUDA_HOST_DEVICE unsigned int func3() { return 12U; }

GET_CUDA_HOST_DEVICE_FUNC(func3)
FUNC_AS_CALLABLE(func3);

DYND_CUDA_HOST_DEVICE double func4(const double (&x)[3], const double (&y)[3])
{
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

GET_CUDA_HOST_DEVICE_FUNC(func4)
FUNC_AS_CALLABLE(func4);

DYND_CUDA_HOST_DEVICE long func5(const long (&x)[2][3])
{
  return x[0][0] + x[0][1] + x[1][2];
}

GET_CUDA_HOST_DEVICE_FUNC(func5)
FUNC_AS_CALLABLE(func5);

DYND_CUDA_HOST_DEVICE int func6(int x, int y, int z) { return x * y - z; }

GET_CUDA_HOST_DEVICE_FUNC(func6)
FUNC_AS_CALLABLE(func6);

DYND_CUDA_HOST_DEVICE double func7(int x, int y, double z)
{
  return (x % y) * z;
}

GET_CUDA_HOST_DEVICE_FUNC(func7)
FUNC_AS_CALLABLE(func7);

#undef GET_HOST_FUNC
#undef GET_CUDA_DEVICE_FUNC
#undef GET_CUDA_HOST_DEVICE_FUNC

#undef FUNC_AS_CALLABLE


class callable0 {
  int m_z;

public:
  DYND_CUDA_HOST_DEVICE callable0(int z = 7) : m_z(z) {}

  DYND_CUDA_HOST_DEVICE int operator()(int x, int y) const
  {
    return 2 * (x - y) + m_z;
  }
};

class callable1 {
  int m_x, m_y;

public:
  DYND_CUDA_HOST_DEVICE callable1(int x, int y) : m_x(x + 2), m_y(y + 3) {}

  DYND_CUDA_HOST_DEVICE int operator()(int z) const { return m_x * m_y - z; }
};

class callable2 {
public:
  CUDA_DEVICE_IF_CUDA_ELSE_HOST double operator()(double x) const
  {
    return 10 * x;
  }
};

TEST(Apply, Function)
{
  typedef Apply<integral_constant<kernel_request_t, kernel_request_host>>
      TestFixture;

  nd::arrfunc af;

  af = nd::make_apply_arrfunc<kernel_request_host, decltype(&func0), &func0>();
  EXPECT_ARR_EQ(TestFixture::To(4), af(TestFixture::To(5), TestFixture::To(3)));

  af = nd::make_apply_arrfunc<kernel_request_host, decltype(&func2), &func2>();
  EXPECT_ARR_EQ(TestFixture::To(13.6f),
                af(TestFixture::To({3.9f, -7.0f, 16.7f})
                       .view(ndt::make_type<float[3]>())));

  af = nd::make_apply_arrfunc<kernel_request_host, decltype(&func3), &func3>();
  EXPECT_ARR_EQ(TestFixture::To(12U), TestFixture::To(af()));

  af = nd::make_apply_arrfunc<kernel_request_host, decltype(&func6), &func6>();
  EXPECT_ARR_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::make_apply_arrfunc<kernel_request_host, decltype(&func7), &func7>();
  EXPECT_ARR_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));
}

TEST(Apply, FunctionWithKeywords)
{
  nd::arrfunc af;

  af = nd::make_apply_arrfunc<decltype(&func0), &func0>("y");
  EXPECT_EQ(4, af(5, dynd::kwds("y", 3)).as<int>());

  af = nd::make_apply_arrfunc(&func0, "y");
  EXPECT_EQ(4, af(5, kwds("y", 3)).as<int>());

  af = nd::make_apply_arrfunc<decltype(&func0), &func0>("x", "y");
  EXPECT_EQ(4, af(5, kwds("x", 5, "y", 3)).as<int>());
//  af = nd::make_apply_arrfunc(&func0, "x", "y");
// EXPECT_EQ(4, af(5, kwds("x", 5, "y", 3)).as<int>());

#ifndef __CUDACC__
/*
  af = nd::make_apply_arrfunc<decltype(&func1), &func1>("y");
  EXPECT_EQ(53.15, af(3.75, kwds("y", 19)).as<double>());
//  af = nd::make_apply_arrfunc(&func1, "y");
  //EXPECT_EQ(53.15, af(3.75, kwds("y", 19)).as<double>());

  af = nd::make_apply_arrfunc<decltype(&func1), &func1>("x", "y");
  EXPECT_EQ(53.15, af(kwds("x", 3.75, "y", 19)).as<double>());
//  af = nd::make_apply_arrfunc(&func1, "x", "y");
//  EXPECT_EQ(53.15, af(kwds("x", 3.75, "y", 19)).as<double>());
*/
#endif

  // TODO: Enable tests with reference types as keywords

  af = nd::make_apply_arrfunc<decltype(&func6), &func6>("z");
  EXPECT_EQ(8, af(3, 5, kwds("z", 7)).as<int>());
  af = nd::make_apply_arrfunc(&func6, "z");
  EXPECT_EQ(8, af(3, 5, kwds("z", 7)).as<int>());

  af = nd::make_apply_arrfunc<decltype(&func6), &func6>("y", "z");
  EXPECT_EQ(8, af(3, kwds("y", 5, "z", 7)).as<int>());
  af = nd::make_apply_arrfunc(&func6, "y", "z");
  EXPECT_EQ(8, af(3, kwds("y", 5, "z", 7)).as<int>());

  af = nd::make_apply_arrfunc<decltype(&func6), &func6>("x", "y", "z");
  EXPECT_EQ(8, af(kwds("x", 3, "y", 5, "z", 7)).as<int>());
  af = nd::make_apply_arrfunc(&func6, "x", "y", "z");
  EXPECT_EQ(8, af(kwds("x", 3, "y", 5, "z", 7)).as<int>());

  af = nd::make_apply_arrfunc<decltype(&func7), &func7>("z");
  EXPECT_EQ(36.3, af(38, 5, kwds("z", 12.1)).as<double>());
  af = nd::make_apply_arrfunc(&func7, "z");
  EXPECT_EQ(36.3, af(38, 5, kwds("z", 12.1)).as<double>());

  af = nd::make_apply_arrfunc<decltype(&func7), &func7>("y", "z");
  EXPECT_EQ(36.3, af(38, kwds("y", 5, "z", 12.1)).as<double>());
  af = nd::make_apply_arrfunc(&func7, "y", "z");
  EXPECT_EQ(36.3, af(38, kwds("y", 5, "z", 12.1)).as<double>());

  af = nd::make_apply_arrfunc<decltype(&func7), &func7>("x", "y", "z");
  EXPECT_EQ(36.3, af(kwds("x", 38, "y", 5, "z", 12.1)).as<double>());
  af = nd::make_apply_arrfunc(&func7, "x", "y", "z");
  EXPECT_EQ(36.3, af(kwds("x", 38, "y", 5, "z", 12.1)).as<double>());
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

#ifdef __CUDACC__

  if (TestFixture::KernelRequest == kernel_request_cuda_device) {
    af = nd::make_apply_arrfunc<kernel_request_cuda_device>(
        get_func1<kernel_request_cuda_device>());
    EXPECT_ARR_EQ(TestFixture::To(53.15),
                  af(TestFixture::To(3.75), TestFixture::To(19)));

    af = nd::make_apply_arrfunc<kernel_request_cuda_device>(
        func1_as_callable<kernel_request_cuda_device>());
    EXPECT_ARR_EQ(TestFixture::To(53.15),
                  af(TestFixture::To(3.75), TestFixture::To(19)));

    af =
        nd::make_apply_arrfunc<kernel_request_cuda_device,
                               func1_as_callable<kernel_request_cuda_device>>();
    EXPECT_ARR_EQ(TestFixture::To(53.15),
                  af(TestFixture::To(3.75), TestFixture::To(19)));
  }

#endif

//  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
  //    get_func2<TestFixture::KernelRequest>());
//  EXPECT_ARR_EQ(TestFixture::To(13.6f),
  //              af(TestFixture::To({3.9f, -7.0f, 16.7f})
    //                   .view(ndt::make_type<float[3]>())));

//  af = nd::make_apply_arrfunc<TestFixture::KernelRequest,
  //                            func2_as_callable<TestFixture::KernelRequest>>();

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
      get_func3<TestFixture::KernelRequest>());
  EXPECT_ARR_EQ(TestFixture::To(12U), TestFixture::To(af()));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
      func3_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARR_EQ(TestFixture::To(12U), TestFixture::To(af()));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest,
                              func3_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARR_EQ(TestFixture::To(12U), TestFixture::To(af()));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
      get_func6<TestFixture::KernelRequest>());
  EXPECT_ARR_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
      func6_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARR_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest,
                              func6_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARR_EQ(TestFixture::To(8),
                af(TestFixture::To(3), TestFixture::To(5), TestFixture::To(7)));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
      get_func7<TestFixture::KernelRequest>());
  EXPECT_ARR_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest>(
      func7_as_callable<TestFixture::KernelRequest>());
  EXPECT_ARR_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest,
                              func7_as_callable<TestFixture::KernelRequest>>();
  EXPECT_ARR_EQ(
      TestFixture::To(36.3),
      af(TestFixture::To(38), TestFixture::To(5), TestFixture::To(12.1)));

//  af = nd::make_apply_arrfunc<TestFixture::KernelRequest, callable0>();
//  EXPECT_EQ(11, af(TestFixture::To(5), TestFixture::To(3)).template as<int>());
 // af = nd::make_apply_arrfunc(callable0());
  //EXPECT_EQ(11, af(5, 3).as<int>());

  /*

    af = nd::make_apply_arrfunc(callable0(4));
    EXPECT_EQ(8, af(5, 3).as<int>());

    if (TestFixture::KernelRequest ==
    KERNREQ_CUDA_DEVICE_IF_CUDA_ELSE_KERNREQ_HOST) {
      af = nd::make_apply_arrfunc<KERNREQ_CUDA_DEVICE_IF_CUDA_ELSE_KERNREQ_HOST,
    callable2>();
      EXPECT_EQ(475.0, af(TestFixture::To(47.5)).template as<double>());
    }
  */
}

TYPED_TEST_P(Apply, CallableWithKeywords)
{
  nd::arrfunc af;

  typedef func_wrapper<kernel_request_host, decltype(&func0), &func0>
      func0_as_callable;

  af = nd::make_apply_arrfunc(func0_as_callable(), "y");
  EXPECT_EQ(4, af(5, kwds("y", 3)).as<int>());

  af = nd::make_apply_arrfunc(func0_as_callable(), "x", "y");
  EXPECT_EQ(4, af(5, kwds("x", 5, "y", 3)).as<int>());

#ifdef __CUDACC__

/*
  typedef func_wrapper<kernel_request_host, decltype(&func1), &func1>
      func1_as_callable;

    af = nd::make_apply_arrfunc(func1_as_callable(), "y");
    EXPECT_EQ(53.15, af(3.75, kwds("y", 19)).as<double>());

    af = nd::make_apply_arrfunc(func1_as_callable(), "x", "y");
    EXPECT_EQ(53.15, af(kwds("x", 3.75, "y", 19)).as<double>());
*/

#endif

  // TODO: Enable tests with reference types as keywords

  typedef func_wrapper<kernel_request_host, decltype(&func6), &func6>
      func6_as_callable;

  af = nd::make_apply_arrfunc(func6_as_callable(), "z");
  EXPECT_EQ(8, af(3, 5, kwds("z", 7)).as<int>());

  af = nd::make_apply_arrfunc(func6_as_callable(), "y", "z");
  EXPECT_EQ(8, af(3, kwds("y", 5, "z", 7)).as<int>());

  af = nd::make_apply_arrfunc(func6_as_callable(), "x", "y", "z");
  EXPECT_EQ(8, af(kwds("x", 3, "y", 5, "z", 7)).as<int>());

  typedef func_wrapper<kernel_request_host, decltype(&func7), &func7>
      func7_as_callable;

  af = nd::make_apply_arrfunc(func7_as_callable(), "z");
  EXPECT_EQ(36.3, af(38, 5, kwds("z", 12.1)).as<double>());

  af = nd::make_apply_arrfunc(func7_as_callable(), "y", "z");
  EXPECT_EQ(36.3, af(38, kwds("y", 5, "z", 12.1)).as<double>());

  af = nd::make_apply_arrfunc(func7_as_callable(), "x", "y", "z");
  EXPECT_EQ(36.3, af(kwds("x", 38, "y", 5, "z", 12.1)).as<double>());

  af = nd::make_apply_arrfunc(callable0(), "y");
  EXPECT_EQ(11, af(5, kwds("y", 3)).as<int>());

  af = nd::make_apply_arrfunc(callable0(), "x", "y");
  EXPECT_EQ(11, af(kwds("x", 5, "y", 3)).as<int>());

  af = nd::make_apply_arrfunc(callable0(4), "y");
  EXPECT_EQ(8, af(5, kwds("y", 3)).as<int>());

  af = nd::make_apply_arrfunc(callable0(4), "x", "y");
  EXPECT_EQ(8, af(kwds("x", 5, "y", 3)).as<int>());

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest, callable0, int>("z");
  EXPECT_EQ(8, af(TestFixture::To(5), TestFixture::To(3),
                  kwds("z", TestFixture::To(4))).template as<int>());

  af = nd::make_apply_arrfunc<TestFixture::KernelRequest, callable1, int, int>(
      "x", "y");
  EXPECT_EQ(28, af(TestFixture::To(2),
                   kwds("x", TestFixture::To(1), "y", TestFixture::To(7)))
                    .template as<int>());
}

REGISTER_TYPED_TEST_CASE_P(Apply, Callable, CallableWithKeywords);

typedef integral_constant<kernel_request_t, kernel_request_host>
    KernelRequestHost;
INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Apply, KernelRequestHost);
#ifdef DYND_CUDA
typedef integral_constant<kernel_request_t, kernel_request_cuda_device>
    KernelRequestCUDADevice;
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Apply, KernelRequestCUDADevice);
#endif

/*
  af = nd::make_apply_arrfunc<decltype(&func2), &func2>();
  float x = af(nd::array({3.9f, -7.0f, 16.3f}).view(ndt::make_type<float[3]>()))
                .as<float>();
  EXPECT_FLOAT_EQ(13.2f, x);
  af = nd::make_apply_arrfunc(&func2);
  EXPECT_FLOAT_EQ(13.2f, af(nd::array({3.9f, -7.0f, 16.3f})
                                .view(ndt::make_type<float[3]>())).as<float>());
*/

/*
  af = nd::make_apply_arrfunc<decltype(&func4), &func4>();
  EXPECT_DOUBLE_EQ(
      166.765,
      af(nd::array({9.14, -2.7, 15.32}).view(ndt::make_type<double[3]>()),
         nd::array({0.0, 0.65, 11.0}).view(ndt::make_type<double[3]>()))
          .as<double>());
  af = nd::make_apply_arrfunc(&func4);
  EXPECT_DOUBLE_EQ(
      166.765,
      af(nd::array({9.14, -2.7, 15.32}).view(ndt::make_type<double[3]>()),
         nd::array({0.0, 0.65, 11.0}).view(ndt::make_type<double[3]>()))
          .as<double>());
*/

  /*
    af = nd::make_apply_arrfunc<decltype(&func5), &func5>();
    EXPECT_EQ(1251L, af(nd::array({{1242L, 23L, -5L}, {925L, -836L,
    -14L}}).view(ndt::make_type<long[2][3]>())).as<long>());
    af = nd::make_apply_arrfunc(&func5);
    EXPECT_EQ(1251L, af(nd::array({{1242L, 23L, -5L}, {925L, -836L,
    -14L}}).view(ndt::make_type<long[2][3]>())).as<long>());
  */

/*
  typedef func_wrapper<kernel_request_host, decltype(&func2), &func2>
func2_as_callable;
  af = nd::make_apply_arrfunc<kernel_request_host, func2_as_callable>();
  EXPECT_FLOAT_EQ(13.2f, af(nd::array({3.9f, -7.0f,
16.3f}).view(ndt::make_type<float[3]>())).as<float>());
  af = nd::make_apply_arrfunc(func2_as_callable());
  EXPECT_FLOAT_EQ(13.2f, af(nd::array({3.9f, -7.0f,
16.3f}).view(ndt::make_type<float[3]>())).as<float>());
*/

/*
  af = nd::make_apply_arrfunc<func4_as_callable>();
  EXPECT_DOUBLE_EQ(166.765, af(nd::array({9.14, -2.7,
  15.32}).view(ndt::make_type<double[3]>()),
    nd::array({0.0, 0.65,
  11.0}).view(ndt::make_type<double[3]>())).as<double>());
  af = nd::make_apply_arrfunc(func4_as_callable());
  EXPECT_DOUBLE_EQ(166.765, af(nd::array({9.14, -2.7,
  15.32}).view(ndt::make_type<double[3]>()),
    nd::array({0.0, 0.65,
  11.0}).view(ndt::make_type<double[3]>())).as<double>());
*/

/*
  af = nd::make_apply_arrfunc<func5_as_callable>();
  EXPECT_EQ(1251L, af(nd::array({{1242L, 23L, -5L}, {925L, -836L,
  -14L}}).view(ndt::make_type<long[2][3]>())).as<long>());
  af = nd::make_apply_arrfunc(func5_as_callable());
  EXPECT_EQ(1251L, af(nd::array({{1242L, 23L, -5L}, {925L, -836L,
  -14L}}).view(ndt::make_type<long[2][3]>())).as<long>());
*/