/* Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/Ldim_kENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "kernel/cpu_isa.h"
#include "inner_product_x86.h"

namespace noobsdnn {
namespace icesword {

template <>
Status Operator<X86, INNERPRODUCT, ET_forward_gemm, DT_FLOAT>::execute(
                                    const std::vector<Tensor<X86> *>& inputs,
                                    std::vector<Tensor<X86> *>& outputs,
                                    ImplParam<X86, INNERPRODUCT>& param) {
    auto matrix_b_ptr = static_cast<const float *>(matrix_b_);
    auto matrix_c_ptr = static_cast<float *>(outputs[0]->mutable_data());
    auto offset_ptr = with_bias ? offset_ : nullptr;

    for (auto i = 0; i < inputs.size(); i++) {
        auto dim_k = inputs[i]->shape()[1];
        auto matrix_a_ptr = inputs[i]->data();

        // c = scale * { op(A) + a_offset_scale * a_offset } * { op(B) + b_offset_scale * b_offset } + beta * c + c_offset
        if (i == 0) {
            if (M == 1 || N == 1) {
                gemm.execute(matrix_a_ptr,         // matrix_a
                             matrix_b_ptr,         // matrix_b
                             matrix_c_ptr,         // matrix_c
                             offset_ptr,           // matrix_oc
                             M,                    // matrix_b -> col_major ? width : hight
                             N,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             offset_mode,          // oc_mode
                             false,                // col_major
                             trans_a,              // trans_a
                             trans_b,              // trans_b
                             false,                // pack_a
                             false,                // pack_b
                             0.f,                  // beta
                             1.f);                 // alpha
            } else {
                gemm.execute(matrix_a_ptr,         // matrix_a
                             matrix_b_pack_[i],    // matrix_b
                             matrix_c_ptr,         // matrix_c
                             offset_ptr,           // matrix_oc
                             M,                    // matrix_b -> col_major ? width : hight
                             N,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             offset_mode,          // oc_mode
                             false,                // col_major
                             trans_a,              // trans_a
                             trans_b,              // trans_b
                             false,                // pack_a
                             true,                 // pack_b
                             0.f,                  // beta
                             1.f);                 // alpha not work
            }
        } else {
            if (M == 1 || N == 1) {
                gemm.execute(matrix_a_ptr,         // matrix_a
                             matrix_b_ptr,         // matrix_b
                             matrix_c_ptr,         // matrix_c m*n
                             offset_ptr,           // matrix_oc
                             M,                    // matrix_b -> col_major ? width : hight
                             N,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             'N',                  // oc_mode
                             false,                // col_major
                             trans_a,              // trans_a
                             trans_b,              // trans_b
                             false,                // pack_a
                             false,                // pack_b
                             1.f,                  // beta
                             1.f);                 // alpha
            } else {
                gemm.execute(matrix_a_ptr,         // matrix_a
                             matrix_b_pack_[i],    // matrix_b
                             matrix_c_ptr,         // matrix_c
                             offset_ptr,           // matrix_oc
                             M,                    // matrix_b -> col_major ? width : hight
                             N,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             'N',                  // oc_mode
                             false,                // col_major
                             trans_a,              // trans_a
                             trans_b,              // trans_b
                             false,                // pack_a
                             true,                 // pack_b
                             1.f,                  // beta
                             1.f);                 // alpha not work
            }
        }
        matrix_b_ptr += N * dim_k;
    }

    if (with_active) {
        #pragma omp parallel for collapse(1) num_threads(thread_num)
        for (int id = 0; id < M * N; ++id) {
            float accept = matrix_c_ptr[id];
            if (accept < 0) {
                matrix_c_ptr[id] = 0;
            }
        }
    }

    return S_Success;
}

template <>
Status Operator<X86, INNERPRODUCT, ET_forward_gemm, DT_INT8>::execute(
                                const std::vector<Tensor<X86> *>& inputs,
                                std::vector<Tensor<X86> *>& outputs,
                                ImplParam<X86, INNERPRODUCT>& param) {
    #define __IP_AXPY_ACT_FUNC(height, width) { \
        if (!with_active) { \
            _Pragma("omp parallel for collapse(1) num_threads(thread_num)") \
            for (auto m = 0; m < height; ++m) { \
                _Pragma("omp simd") \
                for (auto n = 0; n < width; ++n) { \
                    auto matrix_c_index = m * width + n; \
                    matrix_c_ptr[matrix_c_index] = nearbyintf(scale[n] * accept_ptr[matrix_c_index]); \
                } \
            } \
        } else { \
            _Pragma("omp parallel for collapse(1) num_threads(thread_num)") \
            for (auto m = 0; m < height; ++m) { \
                _Pragma("omp simd") \
                for (auto n = 0; n < width; ++n) { \
                    auto matrix_c_index = m * width + n; \
                    auto accept = scale[n] * accept_ptr[matrix_c_index]; \
                    if (accept < 0) { \
                        matrix_c_ptr[matrix_c_index] = 0; \
                    } else { \
                        matrix_c_ptr[matrix_c_index] = nearbyintf(accept); \
                    } \
                } \
            } \
        } \
    }

    auto matrix_c_dtype = outputs[0]->get_dtype();
    auto matrix_b_ptr = static_cast<const int8_t *>(matrix_b_);
    auto offset_ptr = with_bias ? static_cast<const int32_t *>(offset_) : nullptr;
    auto accept_ptr = static_cast<int32_t *>(accept_);

    for (auto i = 0; i < inputs.size(); i++) {
        auto dim_k = inputs[i]->shape()[1];
        auto matrix_a_ptr = inputs[i]->data();

        // dst = active { scale * ( op(A) * op(B) + c_offset ) }
        if (i == 0) {
            /*
                c = beta * op(c)
                  + alpha * { op(A) + a_offset_scale * a_offset } * { op(B) + b_offset_scale * b_offset }
                  + op(c_offset)
            */
            if (N == 1 || M == 1) {
                gemm.execute(matrix_b_ptr,         // matrix_a
                             matrix_a_ptr,         // matrix_b
                             accept_ptr,           // matrix_c
                             offset_ptr,           // matrix_oc
                             N,                    // matrix_b -> col_major ? width : hight
                             M,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             offset_mode,          // oc_mode
                             true,                 // col_major
                             trans_b,              // trans_a
                             trans_a,              // trans_b
                             false,                // pack_a
                             false,                // pack_b
                             0.f,                  // beta
                             1.f);                 // alpha
            } else {
                gemm.execute(matrix_b_pack_[i],    // matrix_a
                             matrix_a_ptr,         // matrix_b
                             accept_ptr,           // matrix_c
                             offset_ptr,           // matrix_oc
                             N,                    // matrix_b -> col_major ? width : hight
                             M,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             offset_mode,          // oc_mode
                             true,                 // col_major
                             trans_b,              // trans_a
                             trans_a,              // trans_b
                             true,                 // pack_a
                             false,                // pack_b
                             0.f,                  // beta
                             1.f);                 // alpha
            }
        } else {
            if (N == 1 || M == 1) {
                gemm.execute(matrix_b_ptr,         // matrix_a
                             matrix_a_ptr,         // matrix_b
                             accept_ptr,           // matrix_c
                             nullptr,              // matrix_oc
                             N,                    // matrix_b -> col_major ? width : hight
                             M,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             'N',                  // oc_mode
                             true,                 // col_major
                             trans_b,              // trans_a
                             trans_a,              // trans_b
                             false,                // pack_a
                             false,                // pack_b
                             1.f,                  // beta
                             1.f);                 // alpha
            } else {
                gemm.execute(matrix_b_pack_[i],    // matrix_a
                             matrix_a_ptr,         // matrix_b
                             accept_ptr,           // matrix_c
                             nullptr,              // matrix_oc
                             N,                    // matrix_b -> col_major ? width : hight
                             M,                    // matrix_a -> col_major ? hight : width
                             dim_k,                // matrix a,b common dim
                             0,                    // offset_a
                             0,                    // offset_b
                             'N',                  // oc_mode
                             true,                 // col_major
                             trans_b,              // trans_a
                             trans_a,              // trans_b
                             true,                 // pack_a
                             false,                // pack_b
                             1.f,                  // beta
                             1.f);                 // alpha
            }
        }

        matrix_b_ptr += N * dim_k;
    }

    if (matrix_c_dtype == DT_FLOAT) {
        auto matrix_c_ptr = static_cast<float *>(outputs[0]->mutable_data());
        __IP_AXPY_ACT_FUNC(M, N);
    } else if (matrix_c_dtype == DT_INT32) {
        auto matrix_c_ptr = static_cast<int32_t *>(outputs[0]->mutable_data());
        __IP_AXPY_ACT_FUNC(M, N);
    } else if (matrix_c_dtype == DT_FLOAT) {
        auto matrix_c_ptr = static_cast<int8_t *>(outputs[0]->mutable_data());
        __IP_AXPY_ACT_FUNC(M, N);
    } else if (matrix_c_dtype == DT_UINT8) {
        auto matrix_c_ptr = static_cast<uint8_t *>(outputs[0]->mutable_data());
        __IP_AXPY_ACT_FUNC(M, N);
    } else {
        return S_UnImplError;
    }

    return S_Success;
}

} // namespace icesword
} // namespace noobsdnn