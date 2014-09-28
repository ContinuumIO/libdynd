//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/take_by_pointer_arrfunc.hpp>
#include <dynd/types/pointer_type.hpp>

using namespace std;
using namespace dynd;

TEST(TakeByPointerArrFunc, Simple) {
   // nd::array a = parse_json("4 * int",
    //    "[0, 1, 2, 3]");
  //  nd::array idx = parse_json("4 * int64",
//        "[2, 1, 0, 3]");

    //nd::arrfunc af = make_take_by_pointer_arrfunc();

  //  nd::array res = nd::empty(4, ndt::make_pointer(ndt::make_type<int>()));
//    std::cout << af.get()->func_proto << std::endl;
//
//  af.call_out(a, idx, res);
   // std::cout << res << std::endl;

//    nd::array b = res;
   // std::cout << b << std::endl;
  
//    ndt::type tp = ndt::make_typevar_dim("M", ndt::make_typevar_dim("N", ndt::make_typevar("T")));
//    ndt::type tp = ndt::make_typevar_exp_dim("M", "N", ndt::make_typevar("T"));
  //  std::cout << tp << std::endl;
//    static ndt::type param_types[2] = {ndt::type("M**N * T"), ndt::type("N * Ix")};
  //  static ndt::type func_proto = ndt::make_funcproto(param_types, ndt::type("R * pointer[T]"));

//    std::cout << func_proto << std::endl;

//    nd::array a = parse_json("3 * 2 * int",
  //      "[[0, 1], [2, 3], [4, 5]]");
    //nd::array idx = parse_

//    std::exit(-1);
}
