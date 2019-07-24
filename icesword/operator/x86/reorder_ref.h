/* Copyright (c) 2018 NoobsHPC Authors, Inc. All Rights Reserved.

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

#ifndef NBHPC_ICESWORD_OPERATOR_REORDER_H
#define NBHPC_ICESWORD_OPERATOR_REORDER_H

#pragma once

#include <vector>

#include "icesword/types.h"

namespace noobshpc{
namespace icesword{


// reorder_hw2wh<OP_dtypeDType>(weight, weight_reorder, N, dim_k);
template<typename dtype>
static inline Status reorder_hw2wh(const void* in, void* out, const size_t w, const size_t h) {
    CHECK_EQ(h * w != 0, true) << "" << "wrong h,w value !";
    auto src = (const dtype *)in;
    auto dst = (dtype *)out;
    #pragma omp parallel for collapse(1)
    for (auto i = 0; i < h; i++) {
        #pragma omp simd
        for (auto j = 0; j < w; j++) {
            dst[j * h + i] = src[i * w + j];
        }
    }
    return S_Success;
}

// reorder_hw2wh<dtype>((const void**)&weight, &weight_reorder, N, dim_k);
template<typename dtype>
static inline Status reorder_hw2wh(const void** in, void** out, const size_t h, const size_t w) {
    CHECK_EQ(h * w != 0, true) << "" << "wrong h,w value !";
    auto src = (const dtype **)in;
    auto dst = (dtype **)out;
    #pragma omp parallel for collapse(1)
    for (auto i = 0; i < h; i++) {
        #pragma omp simd
        for (auto j = 0; j < w; j++) {
            (*dst)[j * h + i] = (*src)[i * w + j];
        }
    }
    return S_Success;
}

} // namespace icesword
} // namespace noobshpc

#endif // NBHPC_ICESWORD_OPERATOR_REORDER_H
