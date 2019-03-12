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

#ifndef NBDNN_ICESWORD_OPERATOR_X86_KERNEL_GEMM_REF_H
#define NBDNN_ICESWORD_OPERATOR_X86_KERNEL_GEMM_REF_H

#pragma once

#include "mkl.h"
#include "icesword/operator/gemm.h"

#include <iostream>

namespace noobsdnn {
namespace icesword {

/*
    row major:
        mem_a: dim_m * dim_k
        mem_b: dim_k * dim_n
        mem_c: dim_m * dim_n
    col major:
        mem_a: dim_k * dim_m
        mem_b: dim_n * dim_k
        mem_c: dim_n * dim_m
    matrix(C) = beta  * matrix(C) + offset_c
              + alpha * { matrix(A) + offset_a } * { op(B) + offset_b }
*/

template<DataType a_dtype,
         DataType b_dtype,
         DataType c_dtype>
class GEMM_REF <X86, a_dtype, b_dtype, c_dtype> {

public:
    typedef typename DataTrait<X86, a_dtype>::Dtype A_DType;
    typedef typename DataTrait<X86, b_dtype>::Dtype B_DType;
    typedef typename DataTrait<X86, c_dtype>::Dtype C_DType;

    GEMM_REF()
        : thread_num(ice_get_max_threads())
     {}
    ~GEMM_REF() {}

    Status execute(const void* a_mem,
                   const void* b_mem,
                   const void* oc_mem,
                         void* c_mem,
                   const char oc_mode,
                   const bool col_major,
                   const size_t oa,
                   const size_t ob,
                   const size_t m,
                   const size_t n,
                   const size_t k,
                   const bool trans_a,
                   const bool trans_b,
                   const float beta,
                   const float alpha) {
        size_t lda, ldb, ldc;
        if (col_major) {
            lda = trans_a ? k : m;
            ldb = trans_b ? n : k;
            ldc = m;
        } else {
            lda = trans_a ? m : k;
            ldb = trans_b ? k : n;
            ldc = n;
        }

        return execute(a_mem, b_mem, oc_mem, c_mem, col_major,
                       oa, ob, m, n, k, lda, ldb, ldc, trans_a,
                       trans_b, beta, alpha, oc_mode);
    }

    Status execute(const void* a_mem,
                   const void* b_mem,
                   const void* oc_mem,
                         void* c_mem,
                   const bool col_major,
                   const size_t oa,
                   const size_t ob,
                   const size_t m,            // mem_c -> col_major ? width : hight
                   const size_t n,            // mem_c -> col_major ? hight : width
                   const size_t k,            // matrix a,b common dim
                   const size_t lda,          // len(mem_a) / m
                   const size_t ldb,          // len(mem_b) / n
                   const size_t ldc,          // len(mem_c) / k
                   const bool trans_a,
                   const bool trans_b,
                   const float beta,
                   const float alpha,
                   const char oc_mode) {
    auto status = execute_check(a_mem, b_mem, oc_mem, c_mem, oc_mode);
    if (status != S_Success) {
        return S_InvalidValue;
    }

    bool a_trans = false;
    bool b_trans = false;
    size_t dim_m = 0;
    size_t dim_n = 0;
    size_t dim_k = 0;
    size_t stride_a = 0;
    size_t stride_b = 0;
    size_t offset_a = 0;
    size_t offset_b = 0;
    size_t oc_method = 1;

    const A_DType * mem_a  = nullptr;
    const B_DType * mem_b  = nullptr;
    const C_DType * mem_o  = nullptr;
          C_DType * mem_c  = nullptr;

    /*
        ROW_MAJOR : 
            mem_a : {m, k}
            mem_b : {k, n}
            mem_c : {m, n}
        COL_MAJOR : 
            mem_a : {k, m}
            mem_b : {n, k}
            mem_c : {n, m}
        Convert to row major 
    */
    if (col_major) {
        mem_a = static_cast<const A_DType *>(b_mem);
        mem_b = static_cast<const B_DType *>(a_mem);
        mem_o = static_cast<const C_DType *>(oc_mem);
        mem_c = static_cast<C_DType *>(c_mem);
        a_trans = trans_b;
        b_trans = trans_a;
        offset_a = ob;
        offset_b = oa;
        dim_m = n;
        dim_n = m;
        dim_k = k;
        stride_a = ldb;
        stride_b = lda;
        oc_method = oc_mode == 'F' ? 2
                  : oc_mode == 'R' ? 4
                  : oc_mode == 'C' ? 3
                  : oc_mode == 'A' ? 5
                  : 1;
    } else {
        mem_a = static_cast<const A_DType *>(a_mem);
        mem_b = static_cast<const B_DType *>(b_mem);
        mem_o = static_cast<const C_DType *>(oc_mem);
        mem_c = static_cast<C_DType *>(c_mem);
        a_trans = trans_a;
        b_trans = trans_b;
        offset_a = oa;
        offset_b = ob;
        dim_m = m;
        dim_n = n;
        dim_k = k;
        stride_a = lda;
        stride_b = ldb;
        oc_method = oc_mode == 'F' ? 2
                  : oc_mode == 'R' ? 3
                  : oc_mode == 'C' ? 4
                  : oc_mode == 'A' ? 5
                  : 1;
    }

    #ifdef ICESWORD_VERBOSE
        // compute as row major, row message
        LOG(INFO) << "GEMM_REF_VERBOSE {"
                  << " transa:" << (a_trans ? "true" : "false")
                  << " transb:" << (b_trans ? "true" : "false")
                  << " m:"      << dim_m
                  << " n:"      << dim_n
                  << " k:"      << dim_k
                  << " oa:"     << offset_a
                  << " ob:"     << offset_b
                  << " lda:"    << stride_a
                  << " ldb:"    << stride_b
                  << " ldc:"    << ldc
                  << " beta:"   << beta
                  << " alpha:"  << alpha
                  << " }";
    #endif

    // compute as row major
    #pragma omp parallel for collapse(2) num_threads(thread_num)
    for (auto m = 0; m < dim_m; ++m) {
        for (auto n = 0; n < dim_n; ++n) {
            C_DType ip_a_b = 0;
            auto c_index = m * ldc + n;
            float mem_c_beta = beta * mem_c[c_index];

            #pragma omp simd
            for (auto k = 0; k < dim_k; ++k) {
                auto ab_index = index_calculate(stride_a, stride_b, m, n, k, a_trans, b_trans);
                auto a_index = ab_index[0];
                auto b_index = ab_index[1];

                ip_a_b += (mem_a[a_index] + offset_a)
                        * (mem_b[b_index] + offset_b);
            }

            float alpha_ab_beta_c = alpha * ip_a_b + mem_c_beta;

            switch (oc_method) {
                case 1 :
                    mem_c[c_index] = alpha_ab_beta_c;
                    break;
                case 2 :
                    mem_c[c_index] = alpha_ab_beta_c + mem_o[0];
                    break;
                case 3 :
                    mem_c[c_index] = alpha_ab_beta_c + mem_o[n];
                    break;
                case 4 :
                    mem_c[c_index] = alpha_ab_beta_c + mem_o[m];
                    break;
            }
        }
    }

    return S_Success;
}

private:
    size_t thread_num;

    Status execute_check(const void* mem_a,
                         const void* mem_b,
                         const void* mem_oc,
                               void* mem_c,
                         const char oc_mode) {
        if (mem_a == nullptr ||
            mem_b == nullptr ||
            mem_b == nullptr) {
            LOG(ERROR) << "wrong empty pointer !";
            return S_InvalidValue;
        }

        if (oc_mode != 'N' &&
            oc_mode != 'F' &&
            oc_mode != 'C' &&
            oc_mode != 'R' &&
            oc_mode != 'A') {
            LOG(ERROR) << "wrong mem_oc mode !";
            return S_InvalidValue;
        }

        if (oc_mode != 'N' && mem_oc == nullptr) {
            LOG(ERROR) << "wrong mem_oc pointer !";
            return S_InvalidValue;
        }

        return S_Success;
    }

    std::vector<size_t> index_calculate(const size_t lda,
                                        const size_t ldb,
                                        const size_t m,
                                        const size_t n,
                                        const size_t k,
                                        const bool trans_a,
                                        const bool trans_b) {
        if (trans_a == false && trans_b == false) {
            // dim_m * dim_k, dim_k * dim_n
            return {m * lda + k, k * ldb + n};
        } else if (trans_a == true && trans_b == false) {
            // dim_k * dim_m, dim_k * dim_n
            return {k * lda + m, k * ldb + n};
        } else if (trans_a == false && trans_b == true) {
            // dim_m * dim_k, dim_n * dim_k
            return {m * lda + k, n * ldb + k};
        } else if (trans_a == true && trans_b == true) {
            // dim_k * dim_m, dim_n * dim_k
            return {k * lda + m, n * ldb + k};
        }
        return {0, 0};
    };

}; // class end

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_X86_KERNEL_GEMM_REF_H