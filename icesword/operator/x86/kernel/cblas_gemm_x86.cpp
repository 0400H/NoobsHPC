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

#include "cblas_gemm_x86.h"

namespace noobsdnn {
namespace icesword {

template <>
void* CBLAS_GEMM<X86, DT_FLOAT>::pack(const void* mem_in,
                                      const bool col_major,
                                      const bool packed_a,
                                      const bool need_trans,
                                      const size_t m,
                                      const size_t n,
                                      const size_t k,
                                      const size_t stride,
                                      const float alpha) {
    CHECK_EQ(mem_in != nullptr, true) << "wrong empty pointer !";

    void* mem_out = nullptr;
    auto layout = col_major ? CblasColMajor : CblasRowMajor;
    auto identifier = packed_a ? CblasAMatrix : CblasBMatrix;
    auto trans = need_trans ? CblasTrans : CblasNoTrans;

    auto length = cblas_gemm_s8u8s32_pack_get_size(identifier, m, n, k);
    mem_out = gmalloc(4 * length, 64);
    cblas_sgemm_pack(layout,
                     identifier,
                     trans,
                     m,
                     n,
                     k,
                     alpha,
                     static_cast<const float *>(mem_in),
                     stride,
                     static_cast<float *>(mem_out));

    return mem_out;
}

template <>
void* CBLAS_GEMM<X86, DT_INT8>::pack(const void* mem_in,
                                     const bool col_major,
                                     const bool packed_a,
                                     const bool need_trans,
                                     const size_t m,
                                     const size_t n,
                                     const size_t k,
                                     const size_t stride,
                                     const float alpha) {
    CHECK_EQ(mem_in != nullptr, true) << "wrong empty pointer !";

    void* mem_out = nullptr;
    auto layout = col_major ? CblasColMajor : CblasRowMajor;
    auto identifier = packed_a ? CblasAMatrix : CblasBMatrix;
    auto trans = need_trans ? CblasTrans : CblasNoTrans;

    auto length = cblas_gemm_s8u8s32_pack_get_size(identifier, m, n, k);
    mem_out = gmalloc(length, 64);
    cblas_gemm_s8u8s32_pack(layout,
                            identifier,
                            trans,
                            m,
                            n,
                            k,
                            mem_in,
                            stride,
                            mem_out);

    return mem_out;
}

template <>
Status CBLAS_GEMM<X86, DT_FLOAT>::execute(const void* mem_a,
                                          const void* mem_b,
                                                void* mem_c,
                                          const void* mem_oc,
                                          const size_t M,
                                          const size_t N,
                                          const size_t K,
                                          const int8_t oa,
                                          const int8_t ob,
                                          const char oc_mode,
                                          const bool col_major,
                                          const bool trans_a,
                                          const bool trans_b,
                                          const bool pack_a,
                                          const bool pack_b,
                                          const float beta,
                                          const float alpha,
                                          const size_t lda,
                                          const size_t ldb,
                                          const size_t ldc) {
    auto status = execute_check(mem_a, mem_b, mem_c,
                                mem_oc, oc_mode, oa, ob);
    if (status != S_Success) {
        return status;
    }

    auto A = static_cast<const float *>(mem_a);
    auto B = static_cast<const float *>(mem_b);
    auto C = static_cast<float *>(mem_c);
    auto OC = static_cast<const float *>(mem_oc);

    auto with_pack = (pack_a || pack_b) ? true : false;
    auto layout = col_major ? CblasColMajor : CblasRowMajor;
    auto a_trans = trans_a ? CblasTrans : CblasNoTrans;
    auto b_trans = trans_b ? CblasTrans : CblasNoTrans;

    if (with_pack) {
        cblas_sgemm_compute(layout,
                            pack_a ? CblasPacked : a_trans,
                            pack_b ? CblasPacked : b_trans,
                            M,
                            N,
                            K,
                            A,
                            lda,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
    } else {
        cblas_sgemm(layout,
                    a_trans,
                    b_trans,
                    M,
                    N,
                    K,
                    alpha,
                    A,
                    lda,
                    B,
                    ldb,
                    beta,
                    C,
                    ldc);
    }

    auto oc_method = 0, oc_h = 0, oc_w = 0;
    if (col_major) {
        oc_h = N;
        oc_w = M;
        oc_method = oc_mode == 'F' ? 1
                  : oc_mode == 'R' ? 3
                  : oc_mode == 'C' ? 2
                  : 0;
        A = static_cast<const float *>(mem_b);
        B = static_cast<const float *>(mem_a);
    } else {
        oc_h = M;
        oc_w = N;
        oc_method = oc_mode == 'F' ? 1
                  : oc_mode == 'R' ? 2
                  : oc_mode == 'C' ? 3 
                  : 0;
    }

    auto thread_num = omp_max_thread;
    if (oc_h <= 2) {
        thread_num = 1;
    } else if (oc_h < omp_max_thread) {
        thread_num = oc_h;
    }

    if (oc_method == 1) {
        if (OC[0] != 0) {
            #pragma omp parallel for collapse(1) num_threads(thread_num)
            for (auto h = 0; h < oc_h; h++) {
                #pragma omp simd
                for (auto w = 0; w < oc_w; w++) {
                    C[h * oc_w + w] += OC[0];
                }
            }
        }
    } else if (oc_method == 2) {
        #pragma omp parallel for schedule(static)
        for (auto h = 0; h < oc_h; h++) {
            cblas_saxpy(oc_w, 1.0, OC, 1.0, C + h * oc_w, 1);
        }
    } else if (oc_method == 3) {
        #pragma omp parallel for collapse(1) num_threads(thread_num)
        for (auto h = 0; h < oc_h; h++) {
            auto oc_tmp = OC[h];
            if (oc_tmp != 0) {
                #pragma omp simd
                for (auto w = 0; w < oc_w; w++) {
                    C[h * oc_w + w] += oc_tmp;
                }
            }
        }
    }

    return S_Success;
}

template <>
Status CBLAS_GEMM<X86, DT_INT8>::execute(const void* mem_a,
                                         const void* mem_b,
                                               void* mem_c,
                                         const void* mem_oc,
                                         const size_t M,
                                         const size_t N,
                                         const size_t K,
                                         const int8_t oa,
                                         const int8_t ob,
                                         const char oc_mode,
                                         const bool col_major,
                                         const bool trans_a,
                                         const bool trans_b,
                                         const bool pack_a,
                                         const bool pack_b,
                                         const float beta,
                                         const float alpha,
                                         const size_t lda,
                                         const size_t ldb,
                                         const size_t ldc) {
    auto status = execute_check(mem_a, mem_b, mem_c,
                                mem_oc, oc_mode, oa, ob);
    if (status != S_Success) {
        return status;
    }

    int32_t zero_c_offset = 0;
    auto offset = static_cast<const int32_t *>(mem_oc);
    auto with_pack = (pack_a || pack_b) ? true : false;
    auto layout = col_major ? CblasColMajor : CblasRowMajor;
    auto a_trans = trans_a ? CblasTrans : CblasNoTrans;
    auto b_trans = trans_b ? CblasTrans : CblasNoTrans;
    auto a_mode = pack_a ? (CBLAS_TRANSPOSE)CblasPacked : a_trans;
    auto b_mode = pack_b ? (CBLAS_TRANSPOSE)CblasPacked : b_trans;
    auto offc_mode = CblasFixOffset;
    if (oc_mode == 'N') {
        offc_mode = CblasFixOffset;
        offset = &zero_c_offset;
    } else if (oc_mode == 'F') {
        offc_mode = CblasFixOffset;
    } else if (oc_mode == 'R') {
        offc_mode = CblasRowOffset;
    } else if (oc_mode == 'C') {
        offc_mode = CblasColOffset;
    }

    if (with_pack) {
        cblas_gemm_s8u8s32_compute(layout,
                                    a_mode,
                                    b_mode,
                                    offc_mode,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    mem_a,
                                    lda,
                                    oa,
                                    mem_b,
                                    ldb,
                                    ob,
                                    beta,
                                    static_cast<int32_t *>(mem_c),
                                    ldc,
                                    offset);
    } else {
        // s8, u8, s32
        cblas_gemm_s8u8s32(layout,
                            a_trans,
                            b_trans,
                            offc_mode,
                            M,
                            N,
                            K,
                            alpha,
                            mem_a,
                            lda,
                            oa,
                            mem_b,
                            ldb,
                            ob,
                            beta,
                            static_cast<int32_t *>(mem_c),
                            ldc,
                            offset);
    }

    return S_Success;
}

} // namespace icesword
} // namespace noobsdnn