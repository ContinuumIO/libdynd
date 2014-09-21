//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/funcproto.hpp>

using namespace std;
using namespace dynd;

TEST(FuncProto, IsConst) {
    EXPECT_TRUE(is_const_funcproto<int ()>::value);

    EXPECT_TRUE(is_const_funcproto<int (int)>::value);
    EXPECT_TRUE(is_const_funcproto<int (const int &)>::value);
    EXPECT_FALSE(is_const_funcproto<int (int &)>::value);

    EXPECT_TRUE(is_const_funcproto<int (int, int)>::value);
    EXPECT_TRUE(is_const_funcproto<int (const int &, float)>::value);
    EXPECT_TRUE(is_const_funcproto<int (int, const float &)>::value);
    EXPECT_FALSE(is_const_funcproto<int (int &, float)>::value);
    EXPECT_FALSE(is_const_funcproto<int (float &, int)>::value);
    EXPECT_FALSE(is_const_funcproto<int (int &, float &)>::value);
    EXPECT_FALSE(is_const_funcproto<int (const int &, float &)>::value);
    EXPECT_FALSE(is_const_funcproto<int (int &, const float &)>::value);

    EXPECT_TRUE(is_const_funcproto<void ()>::value);

    EXPECT_TRUE(is_const_funcproto<void (int &)>::value);

    EXPECT_TRUE(is_const_funcproto<void (int &, int)>::value);
    EXPECT_TRUE(is_const_funcproto<void (int &, float)>::value);
    EXPECT_TRUE(is_const_funcproto<void (int &, const int &)>::value);
    EXPECT_FALSE(is_const_funcproto<void (int &, int &)>::value);
    EXPECT_FALSE(is_const_funcproto<void (int &, float &)>::value);

    EXPECT_TRUE(is_const_funcproto<void (int &, float, const int &)>::value);
    EXPECT_FALSE(is_const_funcproto<void (int &, float &, const int &)>::value);
}
