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

namespace noobshpc {
namespace icesword {

template <>
Status Operator<X86, AXPY, FWD_REF, DT_FLOAT>::execute(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, AXPY>& param) {
    auto alpha = reinterpret_cast<const OP_DType *>(param.get_alpha()->data());
    auto src = reinterpret_cast<const OP_DType *>(inputs[0]->data());
    auto dst = reinterpret_cast<OP_DType *>(outputs[0]->mutable_data());

    int32_t aligned_num = channel / block_size * block_size;

    for (auto b_idx = 0; b_idx < batch; ++b_idx) {
        auto c_idx = 0;
        #pragma simd
        #pragma vector aligned
        for (; c_idx < aligned_num; ++c_idx) {
            auto io_idx = b_idx * channel + c_idx;
            dst[io_idx] += alpha[c_idx] * src[io_idx];
        }
        #pragma simd
        for(; c_idx < channel; ++c_idx) {
            auto io_idx = b_idx * channel + c_idx;
            dst[io_idx] += alpha[c_idx] * src[io_idx];
        }
    }

    return S_Success;
}

} // namespace icesword
} // namespace noobshpc