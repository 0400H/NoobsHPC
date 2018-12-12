/* Copyright (c) 2018 NoobsDNN, Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef NBDNN_ICESWORD_CORE_COMMON_H
#define NBDNN_ICESWORD_CORE_COMMON_H

#include <iostream>
#include <vector>
#include <type_traits>
#include <typeinfo>
#include <stdlib.h>
#include <map>
#include <list>

#include "noobsdnn_config.h"
#include "icesword/types.h"
#include "utils/logger/logger.h"


namespace noobsdnn{

namespace icesword{

#define ICESWORD_CHECK(condition) \
    do { \
    SaberStatus error = condition; \
    CHECK_EQ(error, SaberSuccess) << " " << icesword_get_error_string(error); \
} while (0)

inline const char* icesword_get_error_string(SaberStatus error_code){
    switch (error_code) {
        case SaberSuccess:
            return "NBDNN_ICESWORD_STATUS_SUCCESS";
        case SaberNotInitialized:
            return "NBDNN_ICESWORD_STATUS_NOT_INITIALIZED";
        case SaberInvalidValue:
            return "NBDNN_ICESWORD_STATUS_INVALID_VALUE";
        case SaberMemAllocFailed:
            return "NBDNN_ICESWORD_STATUS_MEMALLOC_FAILED";
        case SaberUnKownError:
            return "NBDNN_ICESWORD_STATUS_UNKNOWN_ERROR";
        case SaberOutOfAuthority:
            return "NBDNN_ICESWORD_STATUS_OUT_OF_AUTHORITH";
        case SaberOutOfMem:
            return "NBDNN_ICESWORD_STATUS_OUT_OF_MEMORY";
        case SaberUnImplError:
            return "NBDNN_ICESWORD_STATUS_UNIMPL_ERROR";
        case SaberWrongDevice:
            return "NBDNN_ICESWORD_STATUS_WRONG_DEVICE";
        default:
            return "NBDNN ICESWORD UNKOWN ERRORS";
    }
}

template <bool If, typename ThenType, typename ElseType>
struct IF {
    /// Conditional type result
    typedef ThenType Type;      // true
};

template <typename ThenType, typename ElseType>
struct IF<false, ThenType, ElseType> {
    typedef ElseType Type;      // false
};

} //namespace icesword

} //namespace noobsdnn

#endif //NBDNN_ICESWORD_CORE_COMMON_H