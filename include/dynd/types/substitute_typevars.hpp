//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_TYPES_TYPE_SUBSTITUTE_HPP
#define DYND_TYPES_TYPE_SUBSTITUTE_HPP

#include <map>

#include <dynd/type.hpp>
#include <dynd/string.hpp>

namespace dynd { namespace ndt {

namespace detail {
  ndt::type internal_substitute(const ndt::type &pattern,
                                const std::map<nd::string, ndt::type> &typevars,
                                bool concrete);
}

/**
 * Substitutes type variables in a pattern type.
 *
 * \param pattern  A symbolic type within which to substitute typevars.
 * \param typevars  A map of names to type var values.
 * \param concrete  If true, requires that the result be concrete.
 */
inline ndt::type substitute(const ndt::type &pattern,
                            const std::map<nd::string, ndt::type> &typevars,
                            bool concrete)
{
  // This check for whether ``pattern`` is symbolic is put here in
  // the inline function to avoid the call overhead in this case
  if (!pattern.is_symbolic()) {
    return pattern;
  } else {
    return detail::internal_substitute(pattern, typevars, concrete);
  }
}

}} // namespace dynd::ndt

#endif // DYND_TYPES_TYPE_SUBSTITUTE_HPP
