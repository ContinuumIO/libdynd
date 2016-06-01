//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "dynd_assertions.hpp"
#include "inc_gtest.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/tuple_type.hpp>

using namespace std;
using namespace dynd;

TEST(TupleType, CreateSimple) {
  ndt::type tp;
  const ndt::tuple_type *tt;

  // Tuple with one field
  tp = ndt::make_type<ndt::tuple_type>({ndt::make_type<int32_t>()});
  EXPECT_EQ(tuple_id, tp.get_id());
  EXPECT_EQ(scalar_kind_id, tp.get_base_id());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_EQ(4u, tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  EXPECT_FALSE(tp.extended<ndt::tuple_type>()->is_variadic());
  EXPECT_EQ(0u, (tp.get_flags() & (type_flag_blockref | type_flag_destructor)));
  tt = tp.extended<ndt::tuple_type>();
  ASSERT_EQ(1, tt->get_field_count());
  EXPECT_EQ(ndt::make_type<int32_t>(), tt->get_field_type(0));
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));

  vector<ndt::type> bases{ndt::make_type<ndt::scalar_kind_type>(), ndt::make_type<ndt::any_kind_type>()};
  EXPECT_EQ(bases, tp.bases());

  // Tuple with two fields
  tp = ndt::make_type<ndt::tuple_type>({ndt::make_type<int16_t>(), ndt::make_type<double>()});
  EXPECT_EQ(tuple_id, tp.get_id());
  EXPECT_EQ(scalar_kind_id, tp.get_base_id());
  EXPECT_EQ(0u, tp.get_data_size());
  EXPECT_EQ((size_t)alignof(double), tp.get_data_alignment());
  EXPECT_FALSE(tp.is_pod());
  EXPECT_FALSE(tp.extended<ndt::tuple_type>()->is_variadic());
  EXPECT_EQ(0u, (tp.get_flags() & (type_flag_blockref | type_flag_destructor)));
  tt = tp.extended<ndt::tuple_type>();
  ASSERT_EQ(2, tt->get_field_count());
  EXPECT_EQ(ndt::make_type<int16_t>(), tt->get_field_type(0));
  EXPECT_EQ(ndt::make_type<double>(), tt->get_field_type(1));
  // Roundtripping through a string
  EXPECT_EQ(tp, ndt::type(tp.str()));
}

TEST(TupleType, Equality) {
  EXPECT_EQ(ndt::type("(int32, float16, int32)"), ndt::type("(int32, float16, int32)"));
  EXPECT_NE(ndt::type("(int32, float16, int32)"), ndt::type("(int32, float16, int32, ...)"));
}

TEST(TupleType, Assign) {
  nd::array a, b;

  a = parse_json("(int32, float64, string)", "[12, 2.5, \"test\"]");
  b = nd::empty("(float64, float32, string)");
  b.vals() = a;
  EXPECT_JSON_EQ_ARR("[12, 2.5, \"test\"]", b);
}

TEST(TupleType, Properties) {
  ndt::type tp =
      ndt::make_type<ndt::tuple_type>({ndt::make_type<int>(), ndt::make_type<dynd::string>(), ndt::make_type<float>()});

  EXPECT_ARRAY_EQ((nd::array{3 * sizeof(size_t), 3 * sizeof(size_t), 3 * sizeof(size_t)}),
                  tp.p<std::vector<uintptr_t>>("metadata_offsets"));
}

TEST(TupleType, IDOf) { EXPECT_EQ(tuple_id, ndt::id_of<ndt::tuple_type>::value); }

TEST(TupleType, Match) {
  ndt::type pattern = ndt::type("(D * int, ...)");
  ndt::type x = ndt::type("22 * 10 * uint64");

  EXPECT_FALSE(pattern.match(x));
}

