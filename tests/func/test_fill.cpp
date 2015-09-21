//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/func/fill.hpp>

using namespace std;
using namespace dynd;


TEST(Fill, ZerosScalar)
{
    nd::array z = nd::zeros(kwds("dst_tp", ndt::type("bool")));
    EXPECT_FALSE(z.as<bool>());
    
}

//TEST(Fill, ZerosOptionScalar) {
//    nd::array z = nd::zeros(kwds("dst_tp", ndt::type("?bool")));
//    EXPECT_FALSE(z.as<bool>());
//}

TEST(Fill, ZerosDims)
{
    nd::array actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * bool")));
    nd::array expected = parse_json("2 * 3 * bool", "[[false, false, false], [false, false, false]]");
    EXPECT_ARRAY_EQ(expected, actual);
}

//TEST(Fill, ZerosOption)
//{
//    nd::array actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * ?bool")));
//    nd::array expected = parse_json("2 * 3 * ?bool", "[[false, false, false], [false, false, false]]");
//    EXPECT_ARRAY_EQ(expected, actual);
//}

TEST(Fill, ZerosSignedIntegers)
{
    nd::array actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * int8")));
    nd::array expected = parse_json("2 * 3 * int8", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * int16")));
    expected = parse_json("2 * 3 * int16", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * int32")));
    expected = parse_json("2 * 3 * int32", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * int64")));
    expected = parse_json("2 * 3 * int64", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
//    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * int128")));
//    expected = parse_json("2 * 3 * int128", "[[0, 0, 0], [0, 0, 0]]");
//    EXPECT_ARRAY_EQ(expected, actual);
}

TEST(Fill, ZerosUnsignedIntegers)
{
    nd::array actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * uint8")));
    nd::array expected = parse_json("2 * 3 * uint8", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * uint16")));
    expected = parse_json("2 * 3 * uint16", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * uint32")));
    expected = parse_json("2 * 3 * uint32", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * uint64")));
    expected = parse_json("2 * 3 * uint64", "[[0, 0, 0], [0, 0, 0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
//    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * uint128")));
//    expected = parse_json("2 * 3 * uint128", "[[0, 0, 0], [0, 0, 0]]");
//    EXPECT_ARRAY_EQ(expected, actual);
}

TEST(Fill, ZerosReals)
{
//    nd::array actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * float16")));
//    nd::array expected = parse_json("2 * 3 * float16", "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]");
//    EXPECT_ARRAY_EQ(expected, actual);
    
    nd::array actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * float32")));
    nd::array expected = parse_json("2 * 3 * float32", "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * float64")));
    expected = parse_json("2 * 3 * float64", "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
//    actual = nd::zeros(kwds("dst_tp", ndt::type("2 * 3 * float128")));
//    expected = parse_json("2 * 3 * float128", "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]");
//    EXPECT_ARRAY_EQ(expected, actual);
}
TEST(Fill, OnesScalar)
{
    nd::array z = nd::ones(kwds("dst_tp", ndt::type("bool")));
    EXPECT_TRUE(z.as<bool>());
    
}

//TEST(Fill, OnesOptionScalar) {
//    nd::array z = nd::ones(kwds("dst_tp", ndt::type("?bool")));
//    EXPECT_FALSE(z.as<bool>());
//}

TEST(Fill, OnesDims)
{
    nd::array actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * bool")));
    nd::array expected = parse_json("2 * 3 * bool", "[[true, true, true], [true, true, true]]");
    EXPECT_ARRAY_EQ(expected, actual);
}

//TEST(Fill, OnesOption)
//{
//    nd::array actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * ?bool")));
//    nd::array expected = parse_json("2 * 3 * ?bool", "[[true, true, true], [true, true, true]]");
//    EXPECT_ARRAY_EQ(expected, actual);
//}

TEST(Fill, OnesSignedIntegers)
{
    nd::array actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * int8")));
    nd::array expected = parse_json("2 * 3 * int8", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * int16")));
    expected = parse_json("2 * 3 * int16", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * int32")));
    expected = parse_json("2 * 3 * int32", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * int64")));
    expected = parse_json("2 * 3 * int64", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
//    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * int128")));
//    expected = parse_json("2 * 3 * int128", "[[1, 1, 1], [1, 1, 1]]");
//    EXPECT_ARRAY_EQ(expected, actual);
}

TEST(Fill, OnesUnsignedIntegers)
{
    nd::array actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * uint8")));
    nd::array expected = parse_json("2 * 3 * uint8", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * uint16")));
    expected = parse_json("2 * 3 * uint16", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * uint32")));
    expected = parse_json("2 * 3 * uint32", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * uint64")));
    expected = parse_json("2 * 3 * uint64", "[[1, 1, 1], [1, 1, 1]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
//    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * uint128")));
//    expected = parse_json("2 * 3 * uint128", "[[1, 1, 1], [1, 1, 1]]");
//    EXPECT_ARRAY_EQ(expected, actual);
}

TEST(Fill, OnesReals)
{
    //    nd::array actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * float16")));
    //    nd::array expected = parse_json("2 * 3 * float16", "[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]");
    //    EXPECT_ARRAY_EQ(expected, actual);
    
    nd::array actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * float32")));
    nd::array expected = parse_json("2 * 3 * float32", "[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * float64")));
    expected = parse_json("2 * 3 * float64", "[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]");
    EXPECT_ARRAY_EQ(expected, actual);
    
    //    actual = nd::ones(kwds("dst_tp", ndt::type("2 * 3 * float128")));
    //    expected = parse_json("2 * 3 * float128", "[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]");
    //    EXPECT_ARRAY_EQ(expected, actual);
}