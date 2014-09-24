//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>

#include "inc_gtest.hpp"
#include "test_memory.hpp"

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class ArithmeticOp : public Memory<T> {
};

TYPED_TEST_CASE_P(ArithmeticOp);

TYPED_TEST_P(ArithmeticOp, SimpleBroadcast) {
    nd::array a, b, c;

    // Two arrays with broadcasting
    const int v0[] = {1,2,3};
    const int v1[2][3] = {{0,1,1}, {2,5,-10}};
    a = TestFixture::To(v0);
    b = TestFixture::To(v1);

    c = (a + b).eval();
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<int>()), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(3, c(1,0).as<int>());
    EXPECT_EQ(7, c(1,1).as<int>());
    EXPECT_EQ(-7, c(1,2).as<int>());
    c = (a - b).eval();
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<int>()), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(1, c(0,1).as<int>());
    EXPECT_EQ(2, c(0,2).as<int>());
    EXPECT_EQ(-1, c(1,0).as<int>());
    EXPECT_EQ(-3, c(1,1).as<int>());
    EXPECT_EQ(13, c(1,2).as<int>());
/*
    c = (b * a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(2, c(0,1).as<int>());
    EXPECT_EQ(3, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(10, c(1,1).as<int>());
    EXPECT_EQ(-30, c(1,2).as<int>());
    c = (b / a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(0, c(0,1).as<int>());
    EXPECT_EQ(0, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(2, c(1,1).as<int>());
    EXPECT_EQ(-3, c(1,2).as<int>());
*/
}

TYPED_TEST_P(ArithmeticOp, StridedScalarBroadcast) {
    nd::array a, b, c;

    // Two arrays with broadcasting
    const int v0[] = {2,4,6};
    a = TestFixture::To(v0);
    b = TestFixture::To(2);

    c = (a + b).eval();
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<int>()), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(8, c(2).as<int>());
    c = (b + a).eval();
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<int>()), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(8, c(2).as<int>());
    c = (a - b).eval();
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<int>()), c.get_dtype());
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(2, c(1).as<int>());
    EXPECT_EQ(4, c(2).as<int>());
    c = (b - a).eval();
    EXPECT_EQ(TestFixture::MakeType(ndt::make_type<int>()), c.get_dtype());
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(-2, c(1).as<int>());
    EXPECT_EQ(-4, c(2).as<int>());
/*
    c = (a * b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(8, c(1).as<int>());
    EXPECT_EQ(12, c(2).as<int>());
    c = (b * a).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(8, c(1).as<int>());
    EXPECT_EQ(12, c(2).as<int>());
    c = (a / b).eval();
    EXPECT_EQ(ndt::make_type<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0).as<int>());
    EXPECT_EQ(2, c(1).as<int>());
    EXPECT_EQ(3, c(2).as<int>());
*/
}

TEST(ArithmeticOp, VarToStridedBroadcast) {
    nd::array a, b, c;

    a = parse_json("2 * var * int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2 * 3 * int32",
                    "[[5, 6, 7], [8, 9, 10]]");

    // VarDim in the first operand
    c = (a + b).eval();
    ASSERT_EQ(ndt::type("strided * strided * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(8, c(0,1).as<int>());
    EXPECT_EQ(10, c(0,2).as<int>());
    EXPECT_EQ(12, c(1,0).as<int>());
    EXPECT_EQ(13, c(1,1).as<int>());
    EXPECT_EQ(14, c(1,2).as<int>());

    // VarDim in the second operand
    c = (b - a).eval();
    ASSERT_EQ(ndt::type("strided * strided * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(4, c(0,0).as<int>());
    EXPECT_EQ(4, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(4, c(1,0).as<int>());
    EXPECT_EQ(5, c(1,1).as<int>());
    EXPECT_EQ(6, c(1,2).as<int>());
}

TEST(ArithmeticOp, VarToVarBroadcast) {
    nd::array a, b, c;

    a = parse_json("2 * var * int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2 * var * int32",
                    "[[5], [6, 7]]");

    // VarDim in both operands, produces VarDim
    c = (a + b).eval();
    ASSERT_EQ(ndt::type("strided * var * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(7, c(0,1).as<int>());
    EXPECT_EQ(8, c(0,2).as<int>());
    EXPECT_EQ(10, c(1,0).as<int>());
    EXPECT_EQ(11, c(1,1).as<int>());

    a = parse_json("2 * var * int32",
                    "[[1, 2, 3], [4]]");
    b = parse_json("2 * 1 * int32",
                    "[[5], [6]]");

    // VarDim in first operand, strided of size 1 in the second
    ASSERT_EQ(ndt::type("strided * var * int32"), c.get_type());
    c = (a + b).eval();
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(7, c(0,1).as<int>());
    EXPECT_EQ(8, c(0,2).as<int>());
    EXPECT_EQ(10, c(1,0).as<int>());

    // Strided of size 1 in the first operand, VarDim in second
    c = (b - a).eval();
    ASSERT_EQ(ndt::type("strided * var * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    EXPECT_EQ(4, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(2, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
}

TYPED_TEST_P(ArithmeticOp, ScalarOnTheRight) {
    nd::array a, b, c;

    const int v0[] = {1,2,3};
    a = TestFixture::To(v0);

    // A scalar on the right
    c = (a + 12).eval();
    EXPECT_EQ(13, c(0).as<int>());
    EXPECT_EQ(14, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (a - 12).eval();
    EXPECT_EQ(-11, c(0).as<int>());
    EXPECT_EQ(-10, c(1).as<int>());
    EXPECT_EQ(-9, c(2).as<int>());
/*
    c = (a * 3).eval();
    EXPECT_EQ(3, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(9, c(2).as<int>());
    c = (a / 2).eval();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(1, c(2).as<int>());
*/
}

TYPED_TEST_P(ArithmeticOp, ScalarOnTheLeft) {
    nd::array a, b, c;

    const int v0[] = {1,2,3};
    a = TestFixture::To(v0);

    // A scalar on the left
    c = ((-1) + a).eval();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(2, c(2).as<int>());
    c = ((-1) - a).eval();
    EXPECT_EQ(-2, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-4, c(2).as<int>());
/*
    c = (5 * a).eval();
    EXPECT_EQ(5, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (-6 / a).eval();
    EXPECT_EQ(-6, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-2, c(2).as<int>());
*/
}

TEST(ArithmeticOp, ComplexScalar) {
    return;

    nd::array a, c;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    a = v0;

    // A complex scalar
    (a + dynd_complex<float>(1, 2)).debug_print(cout);
    c = (a + dynd_complex<float>(1, 2)).eval();
    EXPECT_EQ(dynd_complex<float>(2,2), c(0).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(3,2), c(1).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(4,2), c(2).as<dynd_complex<float> >());
    c = (dynd_complex<float>(0, -1) * a).eval();
    EXPECT_EQ(dynd_complex<float>(0,-1), c(0).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(0,-2), c(1).as<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(0,-3), c(2).as<dynd_complex<float> >());
}

REGISTER_TYPED_TEST_CASE_P(ArithmeticOp, SimpleBroadcast, StridedScalarBroadcast,
    ScalarOnTheRight, ScalarOnTheLeft);

INSTANTIATE_TYPED_TEST_CASE_P(Default, ArithmeticOp, DefaultMemory);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, ArithmeticOp, cuda_device_type);
#endif // DYND_CUDA
