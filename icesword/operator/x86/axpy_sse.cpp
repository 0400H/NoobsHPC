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

#include "axpy.h"
// #include <xmmintrin.h>

namespace noobshpc {
namespace icesword {

template <>
Status Operator<X86, AXPY, FWD_SSE, DT_FLOAT>::execute(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, AXPY>& param) {
    auto alpha = reinterpret_cast<const OP_DType *>(param.get_alpha()->data());
    auto src = reinterpret_cast<const OP_DType *>(inputs[0]->data());
    auto dst = reinterpret_cast<OP_DType *>(outputs[0]->mutable_data());
    int32_t aligned_loop = channel / block_size;
    __m128 x_vec, y_vec, a_vec, res_vec;

    for (auto b_idx = 0; b_idx < batch; ++b_idx) {
        int32_t a_index = 0;
        int32_t xy_index = b_idx * channel;
        for (auto l_idx = 0; l_idx < aligned_loop; ++l_idx) {
            a_vec = _mm_load_ps(alpha + a_index); // Load 4 values
            x_vec = _mm_load_ps(src + xy_index);
            y_vec = _mm_load_ps(dst + xy_index);
            res_vec = _mm_add_ps(_mm_mul_ps(a_vec, x_vec), y_vec);
            _mm_store_ps(dst + xy_index, res_vec); // Store the result
            a_index += block_size;
            xy_index += block_size;
        }
        for(auto idx = a_index; idx < channel; ++idx) {
            a_vec = _mm_load_ss(alpha + a_index);
            x_vec = _mm_load_ss(src + xy_index);
            y_vec = _mm_load_ss(dst + xy_index);
            res_vec = _mm_add_ss(_mm_mul_ss(a_vec, x_vec), y_vec);
            _mm_store_ss(dst + xy_index, res_vec);
            a_index += 1;
            xy_index += 1;
        }
    }

    return S_Success;
}

} // namespace icesword
} // namespace noobshpc