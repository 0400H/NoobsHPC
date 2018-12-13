/* Copyright (c) 2018 NoobsDNN Authors, Inc. All Rights Reserved.

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

#ifndef NBDNN_ICESWORD_UTILS_H
#define NBDNN_ICESWORD_UTILS_H

#include <iostream>
#include <vector>

#include "icesword/types.h"
#include "utils/logger/logger.h"

namespace noobsdnn{
namespace icesword{

#define ICESWORD_CHECK(condition) {\
    Status error = condition; \
    CHECK_EQ(error, S_Success) << " " << icesword_get_error_string(error); \
}

inline const char* icesword_get_error_string(Status error_code){
    switch (error_code) {
        case S_Success:
            return "NBDNN_ICESWORD_STATUS_SUCCESS";
        case S_NotInitialized:
            return "NBDNN_ICESWORD_STATUS_NOT_INITIALIZED";
        case S_InvalidValue:
            return "NBDNN_ICESWORD_STATUS_INVALID_VALUE";
        case S_MemAllocFailed:
            return "NBDNN_ICESWORD_STATUS_MEMALLOC_FAILED";
        case S_UnKownError:
            return "NBDNN_ICESWORD_STATUS_UNKNOWN_ERROR";
        case S_OutOfAuthority:
            return "NBDNN_ICESWORD_STATUS_OUT_OF_AUTHORITH";
        case S_OutOfMem:
            return "NBDNN_ICESWORD_STATUS_OUT_OF_MEMORY";
        case S_UnImplError:
            return "NBDNN_ICESWORD_STATUS_UNIMPL_ERROR";
        default:
            return "NBDNN ICESWORD UNKOWN ERRORS";
    }
}


} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_UTILS_H