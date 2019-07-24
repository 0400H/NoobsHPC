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
Status Operator<X86, ACT, FWD_AVX2, DType>::init(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, ACT>& param) {
    return S_Success;
}

template <DataType DType>
Status Operator<X86, ACT, FWD_AVX2, DType>::execute(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, ACT>& param) {
    const OP_DType *src = nullptr;
    OP_DType *dst = nullptr;

    if (param.algo_act == "relu") {
        // relu: x > 0 ? x :0

    } else if (param.algo_act == "leakyrelu") {
        float scale = param.leakyrelu_scale;

        // relu: x > 0 ? x : w * x

    } else if (param.algo_act == "sigmoid") {
        // sigmoid: 1/(exp(-x) + 1)

    } else if (param.algo_act == "tanh") {
        // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    } else {
        LOG(FATAL) << "unsupport activation function type !";
    }

    return S_Success;
}

template class Operator<X86, ACT, FWD_AVX2, DT_FLOAT>;
// template class Operator<X86, ACT, FWD_AVX2, DT_INT8>;

} // namespace icesword
} // namespace noobshpc