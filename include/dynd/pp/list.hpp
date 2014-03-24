//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__LIST_HPP_
#define _DYND__LIST_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/if.hpp>
#include <dynd/pp/token.hpp>

#define DYND_PP_FLATTEN(A) DYND_PP_ID A

#define DYND_PP_IS_EMPTY(...) DYND_PP__IS_EMPTY(DYND_PP_HAS_COMMA(__VA_ARGS__), \
    DYND_PP_HAS_COMMA(DYND_PP_TO_COMMA __VA_ARGS__), DYND_PP_HAS_COMMA(__VA_ARGS__ ()), \
    DYND_PP_HAS_COMMA(DYND_PP_TO_COMMA __VA_ARGS__ ()))

#define DYND_PP__IS_EMPTY(A, B, C, D) DYND_PP_HAS_COMMA(DYND_PP_CAT_5(DYND_PP__IS_EMPTY_, \
    A, B, C, D))
#define DYND_PP__IS_EMPTY_0000 DYND_PP__IS_EMPTY_0000
#define DYND_PP__IS_EMPTY_0001 ,
#define DYND_PP__IS_EMPTY_0010 DYND_PP__IS_EMPTY_0010
#define DYND_PP__IS_EMPTY_0011 DYND_PP__IS_EMPTY_0011
#define DYND_PP__IS_EMPTY_0100 DYND_PP__IS_EMPTY_0100
#define DYND_PP__IS_EMPTY_0101 DYND_PP__IS_EMPTY_0101
#define DYND_PP__IS_EMPTY_0110 DYND_PP__IS_EMPTY_0110
#define DYND_PP__IS_EMPTY_0111 DYND_PP__IS_EMPTY_0111
#define DYND_PP__IS_EMPTY_1000 DYND_PP__IS_EMPTY_1000
#define DYND_PP__IS_EMPTY_1001 DYND_PP__IS_EMPTY_1001
#define DYND_PP__IS_EMPTY_1010 DYND_PP__IS_EMPTY_1010
#define DYND_PP__IS_EMPTY_1011 DYND_PP__IS_EMPTY_1011
#define DYND_PP__IS_EMPTY_1100 DYND_PP__IS_EMPTY_1100
#define DYND_PP__IS_EMPTY_1101 DYND_PP__IS_EMPTY_1101
#define DYND_PP__IS_EMPTY_1110 DYND_PP__IS_EMPTY_1110
#define DYND_PP__IS_EMPTY_1111 DYND_PP__IS_EMPTY_1111

#define DYND_PP_LEN(...) DYND_PP_IF_ELSE(DYND_PP_IS_EMPTY(__VA_ARGS__))(0)(DYND_PP_LEN_NONZERO(__VA_ARGS__))

#define DYND_PP_GET(INDEX, ...) DYND_PP__GET((INDEX, __VA_ARGS__))
#define DYND_PP__GET(ARGS) DYND_PP___GET(DYND_PP_SLICE_FROM ARGS)
#define DYND_PP___GET(...) DYND_PP_FIRST(__VA_ARGS__)

#define DYND_PP_FIRST(HEAD, ...) HEAD
#define DYND_PP_LAST(...) DYND_PP_GET(DYND_PP_DEC(DYND_PP_LEN(__VA_ARGS__)), __VA_ARGS__)

#define DYND_PP_SLICE(START, STOP, ...) DYND_PP_SLICE_TO(DYND_PP_SUB(STOP, START), \
    DYND_PP_SLICE_FROM(DYND_PP_FLATTEN((START, __VA_ARGS__))))

#define DYND_PP_RANGE(...) DYND_PP_CAT_2(DYND_PP_RANGE_, DYND_PP_LEN(__VA_ARGS__))(__VA_ARGS__)
#define DYND_PP_RANGE_1(STOP) DYND_PP_RANGE_2(0, STOP)
#define DYND_PP_RANGE_2(START, STOP) DYND_PP_SLICE(START, STOP, DYND_PP_INTS)

#endif
