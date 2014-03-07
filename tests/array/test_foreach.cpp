//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/array.hpp>
#include <dynd/foreach.hpp>

using namespace std;
using namespace dynd;

int intfunc(int x, int y)
{
    return 2 * (x - y);
}

TEST(ArrayViews, IntFunc) {
    nd::array a = 10, b = 20, c;

    c = nd::foreach(a, b, intfunc);
    EXPECT_EQ(-20, c.as<int>());
}