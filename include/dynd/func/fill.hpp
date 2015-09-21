//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
    namespace nd {
        
        extern DYND_API struct zeros : declfunc<zeros> {
            static DYND_API callable children[DYND_TYPE_ID_MAX + 1];
            
            static DYND_API callable make();
        } zeros;
        
        extern DYND_API struct ones : declfunc<ones> {
            static DYND_API callable children[DYND_TYPE_ID_MAX + 1];
            
            static DYND_API callable make();
        } ones;
    } // namespace dynd::nd
} // namespace dynde