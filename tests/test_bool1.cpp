//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/config.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class Bool1 : public ::testing::Test {
};

template <typename T>
class Bool : public ::testing::Test {
};

TYPED_TEST_CASE_P(Bool1);
TYPED_TEST_CASE_P(Bool);

TEST(Bool1, DefaultConstructor) { EXPECT_EQ(bool(), bool1()); }

TYPED_TEST_P(Bool1, Conversion)
{
  EXPECT_TRUE(static_cast<TypeParam>(bool1(true)));
  EXPECT_FALSE(static_cast<TypeParam>(bool1(false)));

//  EXPECT_TRUE(static_cast<bool1>(TypeParam(true)));
//  EXPECT_FALSE(static_cast<bool1>(TypeParam(false)));
}

TYPED_TEST_P(Bool, Conversion)
{
  EXPECT_TRUE(static_cast<TypeParam>(bool(true)));
  EXPECT_FALSE(static_cast<TypeParam>(bool(false)));

//  EXPECT_TRUE(static_cast<bool>(TypeParam(true)));
//  EXPECT_FALSE(static_cast<bool>(TypeParam(false)));
}

/*
TEST(Bool1, IsConvertible)
{
    EXPECT_TRUE((is_convertible<bool1, bool>::value));

    EXPECT_TRUE((is_convertible<bool1, int8>::value));
    EXPECT_TRUE((is_convertible<bool1, int16>::value));
    EXPECT_TRUE((is_convertible<bool1, int32>::value));
    EXPECT_TRUE((is_convertible<bool1, int64>::value));
    EXPECT_TRUE((is_convertible<bool1, int128>::value));
    EXPECT_TRUE((is_convertible<bool1, uint8>::value));
    EXPECT_TRUE((is_convertible<bool1, uint16>::value));
    EXPECT_TRUE((is_convertible<bool1, uint32>::value));
    EXPECT_TRUE((is_convertible<bool1, uint64>::value));
    EXPECT_TRUE((is_convertible<bool1, uint128>::value));
    EXPECT_TRUE((is_convertible<bool1, float32>::value));
    EXPECT_TRUE((is_convertible<bool1, float64>::value));
    EXPECT_TRUE((is_convertible<bool1, complex64>::value));
}
*/

typedef ::testing::Types<bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64> types;

REGISTER_TYPED_TEST_CASE_P(Bool1, Conversion);
REGISTER_TYPED_TEST_CASE_P(Bool, Conversion);


INSTANTIATE_TYPED_TEST_CASE_P(BuiltinTypes, Bool1, types);
INSTANTIATE_TYPED_TEST_CASE_P(BuiltinTypes, Bool, types);