//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/func/random.hpp>

typedef testing::Types<int32_t, int64_t, uint32_t, uint64_t> IntegralTypes;
typedef testing::Types<float, double> RealTypes;

using namespace std;
using namespace dynd;

template <typename T>
class Random : public ::testing::Test {
public:
  typedef T DType;
};

TYPED_TEST_CASE_P(Random);

TYPED_TEST_P(Random, Uniform)
{
  typename TestFixture::DType a = 0;
  typename TestFixture::DType b = 10;
  intptr_t size = 10000;

  ndt::type dst_tp =
      ndt::make_fixed_dim(size, ndt::make_type<typename TestFixture::DType>());
  nd::array res = nd::random::uniform(kwds("a", a, "b", b, "dst_tp", dst_tp));

  double mean = 0;
  for (intptr_t i = 0; i < size; ++i) {
    mean += res(i).as<typename TestFixture::DType>();
  }
  mean /= size;

  EXPECT_EQ_RELERR(static_cast<double>(a + b) / 2, mean, 0.1);
}

#if DYND_CUDA
TEST(Random, CUDAUniform)
{
  nd::arrfunc af = static_cast<nd::arrfunc>(nd::random::uniform);
//  std::cout << af << std::endl;

  ndt::type dst_tp = ndt::type("cuda_device[1000 * float64]");
//  std::cout << nd::random::uniform(kwds("dst_tp", dst_tp)) << std::endl;

//  std::cout << ndt::type("(a: ?R, b: ?R, dst_tp: type) -> M[Dims... * R]").matches(
  //  ndt::type("(a: ?R, b: ?R, dst_tp: type) -> Dims... * R")) << std::endl;

}
#endif

REGISTER_TYPED_TEST_CASE_P(Random, Uniform);
INSTANTIATE_TYPED_TEST_CASE_P(Integral, Random, IntegralTypes);
INSTANTIATE_TYPED_TEST_CASE_P(Real, Random, RealTypes);