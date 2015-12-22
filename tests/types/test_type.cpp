//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <complex>
#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/type.hpp>
#include <dynd/types/any_kind_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_dim_kind_type.hpp>

using namespace std;
using namespace dynd;

TEST(Type, NDTTypeObject)
{
  // The ndt::type just contains one ndt::base_type *
  EXPECT_EQ(sizeof(ndt::base_type *), sizeof(ndt::type));
}

TEST(Type, BasicConstructor)
{
  ndt::type d;

  // Default-constructed type properties
  EXPECT_EQ(uninitialized_type_id, d.get_type_id());
  EXPECT_EQ(void_kind, d.get_kind());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(0u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());

  // void type
  d = ndt::type(void_type_id);
  EXPECT_EQ(void_type_id, d.get_type_id());
  EXPECT_EQ(void_kind, d.get_kind());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(0u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // bool type
  d = ndt::type(bool_type_id);
  EXPECT_EQ(bool_type_id, d.get_type_id());
  EXPECT_EQ(bool_kind, d.get_kind());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(1u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int8 type
  d = ndt::type(int8_type_id);
  EXPECT_EQ(int8_type_id, d.get_type_id());
  EXPECT_EQ(sint_kind, d.get_kind());
  EXPECT_EQ(1u, d.get_data_alignment());
  EXPECT_EQ(1u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int16 type
  d = ndt::type(int16_type_id);
  EXPECT_EQ(sint_kind, d.get_kind());
  EXPECT_EQ(2u, d.get_data_alignment());
  EXPECT_EQ(2u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int32 type
  d = ndt::type(int32_type_id);
  EXPECT_EQ(int32_type_id, d.get_type_id());
  EXPECT_EQ(sint_kind, d.get_kind());
  EXPECT_EQ(4u, d.get_data_alignment());
  EXPECT_EQ(4u, d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // int
  d = ndt::make_type<int>();
  EXPECT_EQ(int32_type_id, d.get_type_id());
  EXPECT_EQ(sint_kind, d.get_kind());
  EXPECT_EQ(sizeof(int), d.get_data_alignment());
  EXPECT_EQ(sizeof(int), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // long
  d = ndt::make_type<long>();
  EXPECT_EQ(sint_kind, d.get_kind());
  EXPECT_EQ(sizeof(long), d.get_data_alignment());
  EXPECT_EQ(sizeof(long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // long long
  d = ndt::make_type<long long>();
  EXPECT_EQ(int64_type_id, d.get_type_id());
  EXPECT_EQ(sint_kind, d.get_kind());
  EXPECT_EQ((size_t)alignof(long long), d.get_data_alignment());
  EXPECT_EQ(sizeof(long long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // unsigned int
  d = ndt::make_type<unsigned int>();
  EXPECT_EQ(uint32_type_id, d.get_type_id());
  EXPECT_EQ(uint_kind, d.get_kind());
  EXPECT_EQ(sizeof(unsigned int), d.get_data_alignment());
  EXPECT_EQ(sizeof(unsigned int), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // unsigned long
  d = ndt::make_type<unsigned long>();
  EXPECT_EQ(uint_kind, d.get_kind());
  EXPECT_EQ(sizeof(unsigned long), d.get_data_alignment());
  EXPECT_EQ(sizeof(unsigned long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // unsigned long long
  d = ndt::make_type<unsigned long long>();
  EXPECT_EQ(uint64_type_id, d.get_type_id());
  EXPECT_EQ(uint_kind, d.get_kind());
  EXPECT_EQ((size_t)alignof(unsigned long long), d.get_data_alignment());
  EXPECT_EQ(sizeof(unsigned long long), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // float
  d = ndt::make_type<float>();
  EXPECT_EQ(float32_type_id, d.get_type_id());
  EXPECT_EQ(real_kind, d.get_kind());
  EXPECT_EQ(sizeof(float), d.get_data_alignment());
  EXPECT_EQ(sizeof(float), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // double
  d = ndt::make_type<double>();
  EXPECT_EQ(float64_type_id, d.get_type_id());
  EXPECT_EQ(real_kind, d.get_kind());
  EXPECT_EQ((size_t)alignof(double), d.get_data_alignment());
  EXPECT_EQ(sizeof(double), d.get_data_size());
  EXPECT_TRUE(d.is_builtin());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(Type, TypeIDConstructor)
{
  EXPECT_EQ(bool_type_id, ndt::type(bool_type_id).get_type_id());
  EXPECT_EQ(int8_type_id, ndt::type(int8_type_id).get_type_id());
  EXPECT_EQ(int16_type_id, ndt::type(int16_type_id).get_type_id());
  EXPECT_EQ(int32_type_id, ndt::type(int32_type_id).get_type_id());
  EXPECT_EQ(int64_type_id, ndt::type(int64_type_id).get_type_id());
  EXPECT_EQ(int128_type_id, ndt::type(int128_type_id).get_type_id());
  EXPECT_EQ(uint8_type_id, ndt::type(uint8_type_id).get_type_id());
  EXPECT_EQ(uint16_type_id, ndt::type(uint16_type_id).get_type_id());
  EXPECT_EQ(uint32_type_id, ndt::type(uint32_type_id).get_type_id());
  EXPECT_EQ(uint64_type_id, ndt::type(uint64_type_id).get_type_id());
  EXPECT_EQ(uint128_type_id, ndt::type(uint128_type_id).get_type_id());
  EXPECT_EQ(float16_type_id, ndt::type(float16_type_id).get_type_id());
  EXPECT_EQ(float32_type_id, ndt::type(float32_type_id).get_type_id());
  EXPECT_EQ(float64_type_id, ndt::type(float64_type_id).get_type_id());
  EXPECT_EQ(float128_type_id, ndt::type(float128_type_id).get_type_id());
  EXPECT_EQ(complex_float32_type_id, ndt::type(complex_float32_type_id).get_type_id());
  EXPECT_EQ(complex_float64_type_id, ndt::type(complex_float64_type_id).get_type_id());
  EXPECT_EQ(void_type_id, ndt::type(void_type_id).get_type_id());
  EXPECT_EQ(ndt::bytes_type::make(), ndt::type(bytes_type_id));
  EXPECT_EQ(ndt::fixed_bytes_kind_type::make(), ndt::type(fixed_bytes_type_id));
  EXPECT_EQ(ndt::pointer_type::make(ndt::any_kind_type::make()), ndt::type(pointer_type_id));
}

TEST(TypeTraits, InitializerList)
{
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<int>()), ndt::make_type({0}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<int>()), ndt::make_type({10, -2}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(7, ndt::make_type<int>()), ndt::make_type({0, 1, 2, 3, 4, 5, 6}));

  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<int>())),
            ndt::make_type({{0, 1}, {2, 3}}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::fixed_dim_type>(3, ndt::make_type<int>())),
            ndt::make_type({{0, 1, 2}, {3, 4, 5}}));

  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::var_dim_type>(ndt::make_type<int>())),
            ndt::make_type({{0}, {1, 2}}));
  EXPECT_EQ(ndt::make_type<ndt::fixed_dim_type>(2, ndt::make_type<ndt::var_dim_type>(ndt::make_type<int>())),
            ndt::make_type({{0, 1}, {2}}));
}
