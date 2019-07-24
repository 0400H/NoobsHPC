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

#ifndef NBHPC_ICESWORD_OPERATOR_X86_ACT_H
#define NBHPC_ICESWORD_OPERATOR_X86_ACT_H

#include "icesword/operator/x86/common.h"

namespace noobshpc {
namespace icesword {

// todo: need add param.* as I/O mem to do activation
#define DEFINE_ACT_CLASS(EType) \
    template <DataType DType> \
    class Operator<X86, ACT, EType, DType> \
        : public ImplBase<X86, ImplParam<X86, ACT>> { \
    public:  \
        typedef typename DataTrait<X86, DType>::Dtype OP_DType; \
        typedef ImplBase<X86, ImplParam<X86, ACT>> Impl_t; \
        Operator() { \
            impl = nullptr; \
            block_size = 16; \
            thread_num = ice_get_max_threads(); \
        } \
        ~Operator() { \
            release(); \
        } \
        Status release() { \
            if (impl != nullptr) { \
                delete impl; \
                impl = nullptr; \
            } \
            return S_Success; \
        } \
        Status init(const std::vector<Tensor<X86> *>& inputs, \
                            std::vector<Tensor<X86> *>& outputs, \
                            ImplParam<X86, ACT>& param) override; \
        virtual Status execute(const std::vector<Tensor<X86> *>& inputs, \
                               std::vector<Tensor<X86> *>& outputs, \
                               ImplParam<X86, ACT>& param) override; \
    private: \
        size_t thread_num; \
        size_t block_size; \
        Impl_t* impl; \
    };

// DEFINE_ACT_CLASS(FWD_DEFAULT)
DEFINE_ACT_CLASS(FWD_AVX2)
DEFINE_ACT_CLASS(FWD_REF)

} // namespace icesword
} // namespace noobshpc

#endif // NBHPC_ICESWORD_OPERATOR_X86_ACT_H