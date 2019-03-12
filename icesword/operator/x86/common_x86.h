/*  Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

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

#ifndef ICESWORD_OPERATOR_X86_COMMON_X86_H
#define ICESWORD_OPERATOR_X86_COMMON_X86_H

#pragma once

#include "mkl.h"
#include "icesword/utils.h"
#include "icesword/impl_param.h"
#include "icesword/core/logger/logger.h"
#include "icesword/operator/reorder.h"
#include "icesword/operator/operator.h"
#include "icesword/operator/x86/omp_thread_x86.h"
#include "icesword/operator/x86/kernel/cblas_gemm_x86.h"
#include "icesword/operator/x86/kernel/jit_generate.h"

namespace noobsdnn {
namespace icesword {

} // namespace icesword
} // namespace noobsdnn

#endif // ICESWORD_OPERATOR_X86_COMMON_X86_H
