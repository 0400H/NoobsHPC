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

#include "activation.h"

namespace noobshpc {
namespace icesword {

template <DataType DType>
Status Operator<X86, ACT, FWD_REF, DType>::init(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, ACT>& param) {
    return S_Success;
}

template <DataType DType>
Status Operator<X86, ACT, FWD_REF, DType>::execute(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, ACT>& param) {
    const OP_DType *src = nullptr;
    OP_DType *dst = nullptr;

    if (param.algo_act == "relu") {
        // relu: x > 0 ? x :0
        for (auto num = 0; num < inputs.size(); ++num) {
            auto length = inputs[num]->valid_size();
            src = static_cast<const OP_DType *>(inputs[num]->data());
            dst = static_cast<OP_DType *>(outputs[num]->mutable_data());
            thread_num = (thread_num > length / block_size) ? length / block_size
                       : (thread_num / 4 > length / block_size) ? 1
                       : thread_num;

            #pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto i = 0; i < length; ++i) {
                auto src_data = src[i];
                dst[i] = src_data > 0 ? src_data : 0;
            }

            // optimization not work
            // #pragma omp parallel for collapse(1) num_threads(thread_num)
            // for (auto i = 0; i < length / block_size; ++i) {
            //     #pragma omp simd
            //     for (auto j = 0; j < block_size; ++j) {
            //         auto index = i * block_size + j;
            //         auto src_data = src[index];
            //         dst[index] = src_data > 0 ? src_data : 0;
            //     }
            // }
            // #pragma omp simd
            // for (auto i = length - length % block_size; i < length; ++i) {
            //     auto src_data = src[i];
            //     dst[i] = src_data > 0 ? src_data : 0;
            // }
        }
    } else if (param.algo_act == "leakyrelu") {
        float scale = param.leakyrelu_scale;

        // relu: x > 0 ? x : w * x
        for (auto num = 0; num < inputs.size(); ++num) {
            auto length = inputs[num]->valid_size();
            src = static_cast<const OP_DType *>(inputs[num]->data());
            dst = static_cast<OP_DType *>(outputs[num]->mutable_data());
            thread_num = (thread_num > length / block_size) ? length / block_size
                       : (thread_num / 4 > length / block_size) ? 1
                       : thread_num;

            #pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto i = 0; i < length; ++i) {
                auto src_data = src[i];
                dst[i] = src_data > 0 ? src_data : scale * src_data;
            }
        }
    } else if (param.algo_act == "sigmoid") {
        // sigmoid: 1/(exp(-x) + 1)
        for (auto num = 0; num < inputs.size(); ++num) {
            auto length = inputs[num]->valid_size();
            src = static_cast<const OP_DType *>(inputs[num]->data());
            dst = static_cast<OP_DType *>(outputs[num]->mutable_data());
            thread_num = (thread_num > length / block_size) ? length / block_size
                       : (thread_num / 4 > length / block_size) ? 1
                       : thread_num;

            #pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto i = 0; i < length; ++i) {
                dst[i] = 1.0f / (1.0f + exp(-src[i]));
            }
        }
    } else if (param.algo_act == "tanh") {
        // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        for (auto num = 0; num < inputs.size(); ++num) {
            auto length = inputs[num]->valid_size();
            src = static_cast<const OP_DType *>(inputs[num]->data());
            dst = static_cast<OP_DType *>(outputs[num]->mutable_data());
            thread_num = (thread_num > length / block_size) ? length / block_size
                       : (thread_num / 4 > length / block_size) ? 1
                       : thread_num;

            #pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto i = 0; i < length; ++i) {
                dst[i] = tanh(src[i]);
            }
        }
    } else {
        LOG(FATAL) << "unsupport activation function type !";
    }

    return S_Success;
}

template class Operator<X86, ACT, FWD_REF, DT_FLOAT>;
// template class Operator<X86, ACT, FWD_REF, DT_INT8>;

} // namespace icesword
} // namespace noobshpc
