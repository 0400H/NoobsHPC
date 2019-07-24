/* Copyright (c) 2018 NoobsHPC Authors All Rights Reserve.

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

#ifndef NBHPC_UNITTEST_TEST_COMMONH
#define NBHPC_UNITTEST_TEST_COMMONH

#pragma once

// #define ICESWORD_DEBUG
// #define ICESWORD_VERBOSE

#include "test_func.h"

#include "icesword/types.h"
#include "icesword/utils.h"
#include "icesword/impl_param.h"
#include "icesword/core/logger/logger.h"
#include "icesword/core/tensor/tensor_op.h"
#include "icesword/operator/engine.h"
#include "icesword/operator/operator.h"

using namespace noobshpc::icesword;

template <typename dtype>
static inline int count_diff(const void *src_1, const void *src_2, int lenofmem,
                             double max_ratio = 0, bool print_error = false,
                             bool print_all = false) {
    auto src1 = static_cast<const dtype*>(src_1);
    auto src2 = static_cast<const dtype*>(src_2);

    if (max_ratio <= 0) {
        max_ratio = 5e-3;
    }

    long count = 0;
    // not use omp, top to down
    for (int i = 0; i < lenofmem; ++i) {
        double ratio = fabs(src1[i] - src2[i]) / fabs(src1[i] + src2[i]);
        if (print_all) {
            LOG(INFO) << "{ out["    << i << "]: " << (float)src1[i]
                      << " out_ref[" << i << "]: " << (float)src2[i] << "}";
        }
        if (fabs(src1[i] - src2[i]) <= 1 && fabs(src1[i]) >= 0 && fabs(src1[i]) <= 10) {
            continue;
        }
        if (ratio > max_ratio) {
            ++count;
            if (print_error) {
                LOG(ERROR) << "error { out[" << i << "]: " << (float)src1[i]
                           << " out_ref["    << i << "]: " << (float)src2[i] << "}";
            }
        }
    }

    return count;
}

template <DataType src_dtype>
bool trunc_odd_value(typename DataTrait<X86, src_dtype>::Dtype src) {
    src = trunc(src);
    if (src / 2 == 0) {
        return false;
    } else {
        return true;
    }
}

#endif // NBHPC_UNITTEST_TEST_COMMONH