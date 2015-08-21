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

#include <dynd/func/mean.hpp>

using namespace std;
using namespace dynd;

TEST(Mean, 1D)
{
  EXPECT_ARRAY_EQ(0.0, nd::mean(nd::array{0.0}));
  EXPECT_ARRAY_EQ(1.0, nd::mean(nd::array{1.0}));
  EXPECT_ARRAY_EQ(2.0, nd::mean(nd::array{0.0, 2.0, 4.0}));
  EXPECT_ARRAY_EQ(3.0, nd::mean(nd::array{1.0, 3.0, 5.0}));
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array{0.0, 1.0, 2.0, 3.0, 4.0,
                                          5.0, 6.0, 7.0, 8.0, 9.0}));
}

#if _MSC_VER >= 1900

TEST(Mean, 2D)
{
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array({{0.0, 1.0, 2.0, 3.0, 4.0},
                                           {5.0, 6.0, 7.0, 8.0, 9.0}})));
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array({{9.0, 8.0, 7.0, 6.0, 5.0},
                                           {4.0, 3.0, 2.0, 1.0, 0.0}})));
}

#endif

// template <typename...>
// using void_t = void;

/*
// primary template handles types that have no nested ::type member:
template <class, class = void_t<>>
struct has_type_member : std::false_type {
};

// specialization recognizes types that do have a nested ::type member:
template <class T>
struct has_type_member<T, void_t<typename T::type>> : std::true_type {
};

namespace detail2 {

template <bool Value, typename T, typename... Us>
struct is_common_type_of_2;

template <typename T, typename... Us>
struct is_common_type_of_2<
    true, T, Us...> : std::is_same<T, typename std::common_type<Us...>::type> {
};

template <typename T, typename... Us>
struct is_common_type_of_2<false, T, Us...> : std::false_type {
};
}

template <typename T, typename... Us>
struct is_common_type_of_2
    : detail2::is_common_type_of_2<
          has_type_member<std::common_type<T, Us...>>::value, T, Us...> {
};
*/

TEST(CommonType, X)
{
  EXPECT_FALSE((is_common_type_of<nd::array, void *, nd::array>::value));
  EXPECT_TRUE((is_common_type_of<double, double, int>::value));
}
