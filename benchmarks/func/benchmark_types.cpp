//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <benchmark/benchmark.h>

#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

static void BM_Type_Match(benchmark::State &state)
{
  ndt::type tp("int64");

  while (state.KeepRunning()) {
    tp.match(tp);
  }
}

BENCHMARK(BM_Type_Match);

static void BM_Type_Tuple_Match(benchmark::State &state)
{
  while (state.KeepRunning()) {
    ndt::type("int32");
  }
}

BENCHMARK(BM_Type_Tuple_Match);