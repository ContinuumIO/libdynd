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

#include <dynd/tuple.hpp>

TEST(Array, Tuple)
{
  nd::array a = nd::tuple(5, 4.0, std::initializer_list<int>({10, 12, 24}));
  EXPECT_EQ(a(0).as<int>(), 5);
  EXPECT_EQ(a(1).as<int>(), 4.0);
//  std::cout << a << std::endl;

//  std::exit(-1);
}