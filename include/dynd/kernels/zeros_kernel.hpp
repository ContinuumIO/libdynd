//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>


namespace dynd {
    namespace nd {
        namespace detail {
            template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
            struct zeros_kernel : base_kernel<zeros_kernel<Src0TypeID, Src0TypeKind>, 0> {
                static const size_t data_size = 0;
                typedef typename type_of<Src0TypeID>::type A0;
                
                void single(char *dst, char *const *DYND_UNUSED(src))
                {
                    *reinterpret_cast<A0*>(dst) = static_cast<A0>(0);
                }
            };
            
            template <>
            struct zeros_kernel<bool_type_id, bool_kind> : base_kernel<zeros_kernel<bool_type_id, bool_kind>, 0> {
                static const size_t data_size = 0;
                
                void single(char *dst, char *const *DYND_UNUSED(src))
                {
                    *reinterpret_cast<bool1*>(dst) = false;
                }
            };
            
        }
        
        template <type_id_t Src0TypeID>
        using zeros_kernel = detail::zeros_kernel<Src0TypeID, type_kind_of<Src0TypeID>::value>;
    }
    
    namespace ndt {
        template <type_id_t Src0TypeID>
        struct type::equivalent<nd::zeros_kernel<Src0TypeID>> {
            static type make() {
                return callable_type::make(type(Src0TypeID));
            }
        };
    }
}