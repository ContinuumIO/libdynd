//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/func/apply.hpp>
#include <dynd/func/compound.hpp>

using namespace std;
using namespace dynd;

TEST(Functional, LeftCompound)
{
  nd::callable f = nd::functional::left_compound(
      nd::functional::apply([](int x, int y) { return x + y; }));

  nd::array y = nd::empty(ndt::type::make<int>());
  y.vals() = 3;
  EXPECT_EQ(8, f(nd::array(5), kwds("dst", y)));
}

TEST(Functional, RightCompound)
{
  nd::callable f = nd::functional::right_compound(
      nd::functional::apply([](int x, int y) { return x - y; }));

  nd::array y = nd::empty(ndt::type::make<int>());
  y.vals() = 3;
  EXPECT_EQ(2, f(nd::array(5), kwds("dst", y)));
}
