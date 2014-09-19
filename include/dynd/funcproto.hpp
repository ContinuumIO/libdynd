//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__FUNCPROTO_HPP_
#define _DYND__FUNCPROTO_HPP_

#include <dynd/config.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/meta.hpp>

namespace dynd {

/**
 * A metaprogram that decays a function pointer, or a member function pointer, to
 * a function (proto)type.
 */
template <typename T>
struct funcproto_from;

template <typename R>
struct funcproto_from<R ()> {
    typedef R (type)();
};

#define FUNCPROTO_FROM(N) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_from<R DYND_PP_META_NAME_RANGE(A, N)> { \
        typedef R (type) DYND_PP_META_NAME_RANGE(A, N); \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_FROM, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_FROM

template <typename func_type>
struct funcproto_from<func_type *> {
    typedef typename funcproto_from<func_type>::type type;
};

#define FUNCPROTO_FROM(N) \
    template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_from<R (T::*) DYND_PP_META_NAME_RANGE(A, N)> { \
        typedef R (type) DYND_PP_META_NAME_RANGE(A, N); \
    };  \
\
    template <typename T, typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct funcproto_from<R (T::*) DYND_PP_META_NAME_RANGE(A, N) const> { \
        typedef R (type) DYND_PP_META_NAME_RANGE(A, N); \
    };

DYND_PP_JOIN_MAP(FUNCPROTO_FROM, (), DYND_PP_RANGE(1, DYND_PP_INC(DYND_ARG_MAX)))

#undef FUNCPROTO_FROM

template <typename T>
struct is_const_funcproto_arg {
    static bool const value = std::is_reference<T>::value
        ? std::is_const<typename std::remove_reference<T>::type>::value : true;
};

template <typename funcproto_type>
struct is_const_funcproto;

template <typename R>
struct is_const_funcproto<R ()> {
    static bool const value = true;
};

template <typename R, typename A0>
struct is_const_funcproto<R (A0)> {
    static bool const value = is_const_funcproto_arg<A0>::value;
};

#define IS_CONST_FUNCPROTO(N) \
    template <typename R, DYND_PP_JOIN_MAP_1(DYND_PP_META_TYPENAME, (,), DYND_PP_META_NAME_RANGE(A, N))> \
    struct is_const_funcproto<R DYND_PP_META_NAME_RANGE(A, N)> { \
        static bool const value = is_const_funcproto_arg<A0>::value \
            && is_const_funcproto<R DYND_PP_META_NAME_RANGE(A, 1, N)>::value; \
    };

DYND_PP_JOIN_MAP(IS_CONST_FUNCPROTO, (), DYND_PP_RANGE(2, DYND_PP_INC(DYND_ARG_MAX)))

#undef IS_CONST_FUNCPROTO

} // namespace dynd

#endif // _DYND__FUNCPROTO_HPP_
