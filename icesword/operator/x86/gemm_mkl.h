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

#ifndef NBHPC_ICESWORD_OPERATOR_X86_GEMM_H
#define NBHPC_ICESWORD_OPERATOR_X86_GEMM_H

#include "mkl.h"
#include "icesword/utils.h"
#include "icesword/operator/gemm.h"
#include "icesword/operator/x86/omp_thread.h"

namespace noobshpc {
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
    matrix(C) = alpha * { matrix(A) + offset_a } * { op(B) + offset_b }
                + beta  * matrix(C) + offset_c
*/

template<DataType DType>
class CBLAS_GEMM <X86, DType> {

public:
    typedef typename DataTrait<X86, DType>::Dtype OP_DType;

    CBLAS_GEMM()
        : omp_max_thread(ice_get_max_threads())
    {}

    ~CBLAS_GEMM() {}

    Status release(void* matrix);

    void* pack(const void* matrix,
               const bool col_major,
               const bool pack_a,
               const bool trans,
               const size_t m,
               const size_t n,
               const size_t k,
               const float alpha = 1.f);

    Status execute(const void* mem_a,
                   const void* mem_b,
                         void* mem_c,
                   const void* mem_oc,
                   const size_t m,
                   const size_t n,
                   const size_t k,
                   const int8_t oa,
                   const int8_t ob,
                   const char oc_mode = 'N',
                   const bool col_major = false,
                   const bool trans_a = false,
                   const bool trans_b = false,
                   const bool pack_a = false,
                   const bool pack_b = false,
                   const float beta = 0.f,
                   const float alpha = 1.f);

    void* pack(const void* mem_in,
               const bool col_major,
               const bool pack_a,
               const bool trans,
               const size_t m,
               const size_t n,
               const size_t k,
               const size_t stride,
               const float alpha = 1.f);

    /* c = alpha * { op(A) + a_offset_scale * a_offset }
                 * { op(B) + b_offset_scale * b_offset }
         + beta  * c + c_offset */
    Status execute(const void* mem_a,
                   const void* mem_b,
                         void* mem_c,
                   const void* mem_oc,
                   const size_t m,             // mem_c -> col_major ? width : hight
                   const size_t n,             // mem_c -> col_major ? hight : width
                   const size_t k,             // matrix a,b common dim
                   const int8_t oa,            // offset_a
                   const int8_t ob,            // offset_b
                   const char oc_mode,         // 'N': none, 'F': fix, 'R': row, 'C': col
                   const bool col_major,       // read method: row_major, col_major, same like transpose
                   const bool trans_a,         // mem_a need transpose
                   const bool trans_b,         // mem_b need transpose
                   const bool pack_a,          // a pack optimization
                   const bool pack_b,          // b pack optimization
                   const float beta,           // scale for old C
                   const float alpha,          // scale for compute op(a) * op(b)
                   const size_t lda,           // mem_a width, without any trans or read method
                   const size_t ldb,           // mem_b width, without any trans or read method
                   const size_t ldc);          // mem_c width, without any trans or read method

    Status convert_mem_s82u8(bool exec_it, void* src, size_t length);

    void* compute_offset(bool exec_it, bool trans_b, const int8_t ob,
                         const float alpha, const size_t dim_n,
                         const size_t dim_k, const void* mem_b);

    Status add_offset2mem_c(const bool exec_it, const char oc_mode,
                            const void* mem_in, void* mem_out,
                            const size_t m, const size_t n);

private:
    size_t omp_max_thread;

    Status execute_check(const void* mem_a,
                         const void* mem_b,
                               void* mem_c,
                         const void* mem_oc,
                         const char oc_mode,
                         const int8_t offset_a,
                         const int8_t offset_b);

}; // class end

template<DataType DType>
Status CBLAS_GEMM<X86, DType>::release(void* matrix) {
    if (matrix != nullptr) {
        gfree(matrix);
    }

    return S_Success;
}

template<DataType DType>
void* CBLAS_GEMM<X86, DType>::pack(const void* matrix,
                                   const bool col_major,
                                   const bool pack_a,
                                   const bool trans,
                                   const size_t m,
                                   const size_t n,
                                   const size_t k,
                                   const float alpha) {
    auto lda = 0, ldb = 0, stride = 0;
    if (col_major) {
        lda = trans ? k : m;
        ldb = trans ? n : k;
    } else {
        lda = trans ? m : k;
        ldb = trans ? k : n;
    }
    stride = pack_a ? lda : ldb;

    return pack(matrix, col_major, pack_a,
                trans, m, n, k, stride, alpha);
}

template<DataType DType>
Status CBLAS_GEMM<X86, DType>::execute(const void* mem_a,
                                       const void* mem_b,
                                             void* mem_c,
                                       const void* mem_oc,
                                       const size_t m,
                                       const size_t n,
                                       const size_t k,
                                       const int8_t oa,
                                       const int8_t ob,
                                       const char oc_mode,
                                       const bool col_major,
                                       const bool trans_a,
                                       const bool trans_b,
                                       const bool pack_a,
                                       const bool pack_b,
                                       const float beta,
                                       const float alpha) {
    size_t lda, ldb, ldc, offseta, offsetb;
    if (col_major) {
        lda = trans_a ? k : m;
        ldb = trans_b ? n : k;
        ldc = m;
    } else {
        lda = trans_a ? m : k;
        ldb = trans_b ? k : n;
        ldc = n;
    }

    #ifdef ICESWORD_VERBOSE
        LOG(INFO) << "CBLAS_GEMM_VERBOSE {"
                  << " oc_mode:" << oc_mode
                  << " layout:"  << (col_major ? 'C' : 'R')
                  << " transa:"  << (trans_a ? "true" : "false")
                  << " transb:"  << (trans_b ? "true" : "false")
                  << " m:"       << m
                  << " n:"       << n
                  << " k:"       << k
                  << " oa:"      << int(oa)
                  << " ob:"      << int(ob)
                  << " lda:"     << lda
                  << " ldb:"     << ldb
                  << " ldc:"     << ldc
                  << " beta:"    << beta
                  << " alpha:"   << alpha
                  << " }";
    #endif

    return execute(mem_a, mem_b, mem_c, mem_oc, m, n, k, oa, ob,
                   oc_mode, col_major, trans_a, trans_b, pack_a,
                   pack_b, beta, alpha, lda, ldb, ldc);
}

template<DataType DType>
Status CBLAS_GEMM<X86, DType>::execute_check(const void* mem_a,
                                             const void* mem_b,
                                                   void* mem_c,
                                             const void* mem_oc,
                                             const char oc_mode,
                                             const int8_t offset_a,
                                             const int8_t offset_b) {
    if (mem_a == nullptr ||
        mem_b == nullptr ||
        mem_c == nullptr) {
        LOG(ERROR) << "wrong matrix empty pointer !";
        return S_InvalidValue;
    }

    if (oc_mode != 'N' && mem_oc == nullptr) {
        LOG(ERROR) << "wrong mem_oc pointer !";
        return S_InvalidValue;
    }

    if (oc_mode != 'N' &&
        oc_mode != 'F' &&
        oc_mode != 'C' &&
        oc_mode != 'R') {
        LOG(ERROR) << "wrong mem_oc mode !";
        return S_InvalidValue;
    }

    if (DType == DT_FLOAT && (offset_a != 0 || offset_b != 0)) {
        LOG(ERROR) << "float offset a,b don't support !";
        return S_InvalidValue;
    }

    return S_Success;
}

template<DataType DType>
Status CBLAS_GEMM<X86, DType>::convert_mem_s82u8(bool exec_it, void* src, size_t length) {

    if (exec_it) {
        auto memory = static_cast<uint8_t *>(src);
        if (memory == nullptr) {
            LOG(FATAL) << "wrong empty pointer !";
            return S_InvalidValue;
        }
        #pragma omp parallel for collapse(1)
        for (auto i = 0; i < length; i++) {
            memory[i] += 128;
        }
    }

    return S_Success;
}

template<DataType DType>
void* CBLAS_GEMM<X86, DType>::compute_offset(bool exec_it, bool trans_b, const int8_t ob,
                                             const float alpha, const size_t dim_n,
                                             const size_t dim_k, const void* mem_b) {
    if (exec_it) {
        if (mem_b == nullptr) {
            LOG(FATAL) << "wrong empty pointer !";
            return nullptr;
        }

        auto dst = static_cast<int32_t*>(calloc(dim_n, sizeof(int32_t)));
        auto b_mem = static_cast<const int8_t*>(mem_b);
        auto scale = alpha * -128;
        auto thread_num = omp_max_thread;
        if (dim_n <= 2) {
            thread_num = 1;
        } else if (dim_n < omp_max_thread) {
            thread_num = dim_n;
        }

        if (trans_b) {
            #pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto i = 0; i < dim_n; i++) {
                int32_t b_dim_k_sum = 0;
                #pragma omp simd
                for (auto j = 0; j < dim_k; j++) {
                    b_dim_k_sum += b_mem[i * dim_k + j] + ob;
                }
                dst[i] += scale * b_dim_k_sum;
            }
        } else {
            for (auto i = 0; i < dim_k; i++) {
                #pragma omp parallel for collapse(1) num_threads(thread_num)
                for (auto j = 0; j < dim_n; j++) {
                    dst[j] += scale * (b_mem[i * dim_n + j] + ob);
                }
            }
        }

        return dst;
    }

    return nullptr;
}

template<DataType DType>
Status CBLAS_GEMM<X86, DType>::add_offset2mem_c(const bool exec_it, const char oc_mode,
                                                const void* mem_in, void* mem_out,
                                                const size_t dim_m, const size_t dim_n) {
    if (exec_it && oc_mode == 'C') {
        if (mem_in == nullptr || mem_out == nullptr) {
            LOG(FATAL) << "wrong empty pointer !";
            return S_InvalidValue;
        }

        auto src = static_cast<const int32_t *>(mem_in);
        auto dst = static_cast<int32_t *>(mem_out);

        auto thread_num = omp_max_thread;
        if (dim_m <= 2) {
            thread_num = 1;
        } else if (dim_m < omp_max_thread) {
            thread_num = dim_m;
        }

        #pragma omp parallel for collapse(1) num_threads(thread_num)
        for (auto h = 0; h < dim_m; h++) {
            #pragma omp simd
            for (auto w = 0; w < dim_n; w++) {
                dst[h * dim_n + w] += src[w];
            }
        }
    }

    return S_Success;
}

} // namespace icesword
} // namespace noobshpc

#endif // NBHPC_ICESWORD_OPERATOR_X86_GEMM_H