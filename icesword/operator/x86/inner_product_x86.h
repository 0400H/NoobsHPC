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

#ifndef NBDNN_ICESWORD_OPERATOR_X86_INNER_PRODUCT_H
#define NBDNN_ICESWORD_OPERATOR_X86_INNER_PRODUCT_H

#include "icesword/operator/x86/common_x86.h"

namespace noobsdnn {
namespace icesword {

template <ExecuteMethod EType, DataType DType>
class Operator<X86, INNERPRODUCT, EType, DType>
    : public OperatorBase<X86, ImplParam<X86, INNERPRODUCT>> {

public:
    typedef typename DataTrait<X86, DType>::Dtype OP_DType;

    Operator()
        : accept_(nullptr)
        , offset_(nullptr)
        , matrix_b_(nullptr)
        , thread_num(ice_get_max_threads())
    {}

    ~Operator() {
        release();
    }

    Status init(const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, INNERPRODUCT>& param) override;

    Status execute(const std::vector<Tensor<X86> *>& inputs,
                   std::vector<Tensor<X86> *>& outputs,
                   ImplParam<X86, INNERPRODUCT>& param) override;

    Status release() override;

private:
    CBLAS_GEMM<X86, DType> gemm;

    bool trans_a;
    bool trans_b;
    bool with_bias;
    bool with_active;

    size_t M;
    size_t N;
    size_t TOTAL_K;
    size_t thread_num;
    char offset_mode;
    std::vector<float> scale;

    void * matrix_b_;
    void * offset_;
    void * accept_;
    std::vector<OP_DType *> matrix_b_pack_;

    Status init_check(const std::vector<Tensor<X86> *>& inputs,
                      std::vector<Tensor<X86> *>& outputs,
                      ImplParam<X86, INNERPRODUCT>& param) override;

    Status init_source(const std::vector<Tensor<X86> *>& inputs,
                       std::vector<Tensor<X86> *>& outputs,
                       ImplParam<X86, INNERPRODUCT>& param) override;

};

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, INNERPRODUCT, EType, DType>::release() {
    for (auto & mem : matrix_b_pack_) {
        if (mem) {
            gemm.release(mem);
            mem = nullptr;
        }
    }
    if (accept_) {
        gfree(accept_);
        accept_ = nullptr;
    }
    matrix_b_ = nullptr;
    offset_ = nullptr;

    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, INNERPRODUCT, EType, DType>::init(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, INNERPRODUCT>& param) {
    if (init_check(inputs, outputs, param) != S_Success) {
        return S_UnImplError;
    }
    if (init_source(inputs, outputs, param) != S_Success) {
        return S_UnImplError;
    }
    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, INNERPRODUCT, EType, DType>::init_check(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, INNERPRODUCT>& param) {
    if (inputs.size() == 0 ||
        outputs.size() == 0 ||
        inputs[0] == nullptr ||
        outputs[0] == nullptr ||
        inputs[0]->data() == nullptr ||
        outputs[0]->data() == nullptr ||
        param.get_matrix_b() == nullptr ||
        param.get_matrix_b()->data() == nullptr) {
        LOG(ERROR) << "wrong empty pointer !";
        return S_InvalidValue;
    }

    if (DType == DT_INT8) {
        if (inputs[0]->get_scale().size() == 0 ||
            outputs[0]->get_scale().size() == 0  ||
            param.get_matrix_b()->get_scale().size() == 0 ) {
            LOG(ERROR) << "wrong scale size !";
            return S_InvalidValue;
        }
    }

    M = inputs[0]->shape()[0];
    N = outputs[0]->shape()[1];
    TOTAL_K = param.get_matrix_b()->shape()[1];

    trans_a = param.trans_a;
    trans_b = param.trans_b;
    with_active = param.with_active;
    with_bias = param.get_matrix_bias() ? true : false;
    offset_mode = !with_bias ? 'N'
                : (DType == DT_FLOAT) ? 'R'
                : 'C';

    matrix_b_ = param.get_matrix_b()->data();
    offset_ = with_bias ? param.get_matrix_bias()->data() : nullptr;

    thread_num = thread_num <= M * N
               ? thread_num : M * N;

    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, INNERPRODUCT, EType, DType>::init_source(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, INNERPRODUCT>& param) {
    if (DType == DT_FLOAT) {
        auto total_k = 0;
        for (auto i = 0; i < inputs.size(); i++) {
            auto dim_k = inputs[i]->shape()[1];
            auto B_ = static_cast<const OP_DType *>(matrix_b_) + N * total_k;
            if (M != 1 && N != 1) {
                matrix_b_pack_.push_back(static_cast<OP_DType *>(gemm.pack(B_,          // ptr
                                                                          false,       // col_major
                                                                          false,       // packed_a
                                                                          trans_b,     // need_trans
                                                                          M,           // m
                                                                          N,           // n
                                                                          dim_k,       // k
                                                                          1.f)));      // alpha
            }
            total_k += dim_k;
        }
    } else {
        accept_ = gcalloc(M * N, sizeof(int32_t));
        CHECK_EQ((accept_ != nullptr), true) << "calloc memory failed !";

        auto total_k = 0;
        for (auto i = 0; i < inputs.size(); i++) {
            auto dim_k = inputs[i]->shape()[1];
            auto B_ = static_cast<const OP_DType *>(matrix_b_) + N * total_k;
            if (M != 1 && N != 1) {
                matrix_b_pack_.push_back(static_cast<OP_DType *>(gemm.pack(B_,          // ptr
                                                                           true,        // col_major
                                                                           true,        // packed_a
                                                                           trans_b,     // need_trans
                                                                           N,           // m
                                                                           M,           // n
                                                                           dim_k,       // k
                                                                           1.f)));      // alpha
            }
            total_k += dim_k;
        }

        for (auto i = 0; i < N; i ++) {
            scale.push_back((inputs[0]->get_scale()[0] * param.get_matrix_b()->get_scale()[i]) /
                            outputs[0]->get_scale()[0]);
        }
    }

    return S_Success;
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_X86_INNER_PRODUCT_H