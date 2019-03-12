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

#ifndef NBDNN_ICESWORD_OPERATOR_X86_ACTIVATION_H
#define NBDNN_ICESWORD_OPERATOR_X86_ACTIVATION_H

#include "icesword/operator/x86/common_x86.h"

namespace noobsdnn {
namespace icesword {

// todo: need add param.* as I/O mem to do activation
#define DEFINE_ACTIVATION_CLASS(EType) \
    template <DataType DType> \
    class Operator<X86, ACTIVATION, EType, DType> \
        : public OperatorBase<X86, ImplParam<X86, ACTIVATION>> { \
    public:  \
        typedef typename DataTrait<X86, DType>::Dtype OP_DType; \
        Operator() { \
            block_size = 16; \
            thread_num = ice_get_max_threads(); \
        } \
        ~Operator() { \
            release(); \
        } \
        Status release() { \
            return S_Success; \
        } \
        Status init(const std::vector<Tensor<X86> *>& inputs, \
                    std::vector<Tensor<X86> *>& outputs, \
                    ImplParam<X86, ACTIVATION>& param) { \
            return S_Success; \
        }; \
        Status execute(const std::vector<Tensor<X86> *>& inputs, \
                       std::vector<Tensor<X86> *>& outputs, \
                       ImplParam<X86, ACTIVATION>& param) override; \
    private: \
        size_t thread_num; \
        size_t block_size; \
    };

DEFINE_ACTIVATION_CLASS(ET_forward_gemm)
DEFINE_ACTIVATION_CLASS(ET_forward_jit)

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_X86_ACTIVATION_H