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

#ifndef NBHPC_ICESWORD_OPERATOR_X86_AXPY_H
#define NBHPC_ICESWORD_OPERATOR_X86_AXPY_H

#include "icesword/operator/x86/common.h"

namespace noobshpc {
namespace icesword {

template <ExecuteMethod EType, DataType DType>
class Operator<X86, AXPY, EType, DType>
    : public ImplBase<X86, ImplParam<X86, AXPY>> {
public:
    typedef typename DataTrait<X86, DType>::Dtype OP_DType;
    Operator()
        : block_size(get_block_size(EType))
        , thread_num(ice_get_max_threads())
    {}

    ~Operator() {
        release();
    }

    Status release() {
        return S_Success;
    }

    Status init(const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, AXPY>& param) {
        batch = inputs[0]->shape()[0];
        channel = inputs[0]->shape()[1];

        if (param.get_bias() != nullptr) {
            auto bias = reinterpret_cast<OP_DType *>(param.get_bias()->data());
            auto dst = reinterpret_cast<OP_DType *>(outputs[0]->mutable_data());

            for (auto b_idx = 0; b_idx < batch; ++b_idx) {
                for (auto c_idx = 0; c_idx < channel; ++c_idx) {
                    dst[b_idx * channel + c_idx] += bias[c_idx];
                }
            }
        }

        return S_Success;
    };

    Status execute(const std::vector<Tensor<X86> *>& inputs,
                    std::vector<Tensor<X86> *>& outputs,
                    ImplParam<X86, AXPY>& param) override;
private:
    size_t thread_num;
    size_t block_size;
    size_t batch;
    size_t channel;
};

} // namespace icesword
} // namespace noobshpc

#endif // NBHPC_ICESWORD_OPERATOR_X86_AXPY_H