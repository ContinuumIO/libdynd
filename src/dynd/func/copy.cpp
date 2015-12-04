//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/copy.hpp>
#include <dynd/kernels/copy_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::copy::make() { return nd::callable::make<copy_kernel>(); }

DYND_API struct nd::copy nd::copy;
