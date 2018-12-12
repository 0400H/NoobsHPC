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

#include "icesword/types.h"
#include "icesword/params.h"
#include "icesword/core/context.h"
#include "icesword/core/tensor.h"
#include "icesword/core/device.h"

namespace noobsdnn{

namespace icesword{
    #ifdef USE_OPENMP
        #include <omp.h>
    #endif //openmp

} //namespace icesword
} //namespace noobsdnn

#endif //NBDNN_ICESWORD_CORE_COMMON_H