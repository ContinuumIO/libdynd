//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/fixed_dim_type.hpp>

using namespace std;
using namespace dynd;

std::string ndt::base_string_type::get_utf8_string(const char *arrmeta, const char *data,
                                                   assign_error_mode errmode) const
{
  const char *begin, *end;
  get_string_range(&begin, &end, arrmeta, data);
  return string_range_as_utf8_string(get_encoding(), begin, end, errmode);
}

size_t ndt::base_string_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const { return 0; }

std::map<std::string, std::pair<ndt::type, const char *>> ndt::base_string_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, const char *>> properties;
  properties["encoding"] = {ndt::type("string"), reinterpret_cast<const char *>(&m_encoding_repr)};

  return properties;
}
