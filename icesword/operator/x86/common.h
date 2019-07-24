/*  Copyright (c) 2018 NoobsHPC Authors All Rights Reserve.

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

#ifndef ICESWORD_OPERATOR_X86_COMMON_H
#define ICESWORD_OPERATOR_X86_COMMON_H

#pragma once

#include "mkl.h"
#include "icesword/utils.h"
#include "icesword/types.h"
#include "icesword/impl_param.h"
#include "icesword/core/logger/logger.h"
#include "icesword/operator/operator.h"
#include "icesword/operator/x86/gemm_mkl.h"
#include "icesword/operator/x86/omp_thread.h"
#include "icesword/operator/x86/reorder_ref.h"

namespace noobshpc {
namespace icesword {

static inline int32_t get_block_size(ExecuteMethod em) {
    switch (em) {
        case ET_invalid:
            return 0;
        case FWD_DEFAULT:
            return 8;
        case FWD_REF:
            return 8;
        case FWD_GEMM:
            return 8;
        case FWD_SSE:
            return 4;
        case FWD_AVX2:
            return 8;
    }
}

template <bool, typename, bool, typename, typename> struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> {
    typedef T type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> {
    typedef FT type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> {
    typedef FF type;
};

} // namespace icesword
} // namespace noobshpc

#endif // ICESWORD_OPERATOR_X86_COMMON_H
