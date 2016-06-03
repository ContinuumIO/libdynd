//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/type.hpp>
#include <dynd/types/escape_option_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>

using namespace std;
using namespace dynd;

TEST(EscapeOptionType, Constructor) {
  ndt::type esc_option_tp =
      ndt::make_type<ndt::escape_option_type>(ndt::make_type<ndt::fixed_dim_type>(10, ndt::make_type<double>()));
  EXPECT_EQ(escape_option_id, esc_option_tp.get_id());
  EXPECT_EQ(any_kind_id, esc_option_tp.get_base_id());
  EXPECT_EQ(0u, esc_option_tp.get_data_size());
  EXPECT_EQ(1u, esc_option_tp.get_data_alignment());
  EXPECT_FALSE(esc_option_tp.is_expression());
  EXPECT_TRUE(esc_option_tp.is_symbolic());
  EXPECT_EQ(esc_option_tp, ndt::type(esc_option_tp.str())); // Round trip through a string
}

TEST(EscapeOptionType, Match) {
  EXPECT_TRUE(ndt::type("\\?10 * int32").match(ndt::type("10 * int32")));
  EXPECT_TRUE(ndt::type("\\?10 * int32").match(ndt::type("int32")));

  EXPECT_TRUE(ndt::type("\\?Fixed * Scalar").match(ndt::type("10 * int32")));
  EXPECT_TRUE(ndt::type("\\?Fixed * Scalar").match(ndt::type("Fixed * int32")));
  EXPECT_FALSE(ndt::type("\\?Fixed * Scalar").match(ndt::type("var * int32")));
  EXPECT_FALSE(ndt::type("\\?Fixed * float64").match(ndt::type("10 * int32")));
}
