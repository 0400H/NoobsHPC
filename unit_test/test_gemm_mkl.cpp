/* Copyright (c) 2018 noobsDnn Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICEnSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRAnTIES OR COnDITIOnS OF AnY kInD, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "test_common.h"

template <DataType a_dtype,
          DataType b_dtype,
          DataType c_dtype>
Status test_gemm_cpu(const size_t m, const size_t n, const size_t k,
                     const int8_t offset_a, const int8_t offset_b,
                     const char oc_mode, const bool col_major,
                     const bool trans_a, const bool trans_b,
                     const bool pack_a, const bool pack_b,
                     const float alpha, const float beta) {
    typedef typename DataTrait<X86, a_dtype>::Dtype A_DType;
    typedef typename DataTrait<X86, b_dtype>::Dtype B_DType;
    typedef typename DataTrait<X86, c_dtype>::Dtype C_DType;

    auto io_dtype = get_io_dtype_string(a_dtype, c_dtype);
    auto b_int8 = b_dtype == DT_FLOAT ? false : true;

    size_t lda, ldb, ldc;
    auto oc_size = oc_mode == 'F' ? 1
                 : oc_mode == 'R' ? n
                 : oc_mode == 'C' ? m
                 : 0;

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
        LOG(INFO) << "Test Gemm x86 {"
                  << " io_dtype:" << io_dtype
                  << " layout:"   << (col_major ? "col" : "row")
                  << " oc_mode:"  << oc_mode
                  << " m:"        << m
                  << " n:"        << n
                  << " k:"        << k
                  << " oa:"       << int32_t(offset_a)
                  << " ob:"       << int32_t(offset_b)
                  << " lda:"      << lda
                  << " ldb:"      << ldb
                  << " ldc:"      << ldc
                  << " beta:"     << beta
                  << " alpha:"    << alpha
                  << " pack_a:"   << (pack_a ? "true" : "false")
                  << " pack_b:"   << (pack_b ? "true" : "false")
                  << " trans_a:"  << (trans_a ? "true" : "false")
                  << " trans_b:"  << (trans_b ? "true" : "false")
                  << " }";
    #endif

    CBLAS_GEMM<X86, b_dtype> gemm;
    GEMM_REF<X86, a_dtype, b_dtype, c_dtype> gemm_ref;
    Tensor<X86> tensor_a, tensor_b, tensor_oc, tensor_c, tensor_a_ref, tensor_c_ref;

    // according to sizeof matrix, not follow shape
    Shape A_Shape({m * k}, LT_C);
    Shape B_Shape({n * k}, LT_C);
    Shape OC_Shape({oc_size}, LT_C);
    Shape C_Shape({m * n}, LT_C);

    tensor_a.re_alloc(A_Shape, a_dtype);
    tensor_b.re_alloc(B_Shape, b_dtype);
    tensor_oc.re_alloc(OC_Shape, c_dtype);
    tensor_c.re_alloc(C_Shape, c_dtype);

    tensor_a_ref.re_alloc(A_Shape, a_dtype);
    tensor_c_ref.re_alloc(C_Shape, c_dtype);

    #ifdef ICESWORD_DEBUG
        fill_tensor_debug<a_dtype>(tensor_a.mutable_data(), k, m, true, false);
        fill_tensor_const(tensor_a, 1);
        fill_tensor_const(tensor_b, 1);
        fill_tensor_const(tensor_c, 100);
        fill_tensor_const(tensor_oc, 0);
        fill_tensor_rand(tensor_oc, -128, 127);
    #else
        if (a_dtype == DT_INT8) {
            fill_tensor_rand(tensor_a, -128, 127);
        } else {
            fill_tensor_rand(tensor_a, 0, 255);
        }
        fill_tensor_rand(tensor_b, -128, 127);
        fill_tensor_rand(tensor_oc, -128, 127);
        fill_tensor_rand(tensor_c, -128, 127);
    #endif

    tensor_a_ref.copy_from(tensor_a);
    tensor_c_ref.copy_from(tensor_c);

    auto s8_a = a_dtype == DT_INT8 ? true : false;
    auto mem_oc_s8a = gemm.compute_offset(s8_a, trans_b, offset_b, alpha, n, k, tensor_b.data());
    gemm.convert_mem_s82u8(s8_a, tensor_a.mutable_data(), m * k);

    // compute lda, ldb, ldc by user
    auto mem_a_pack = gemm.pack(tensor_a.data(), col_major, true, trans_a, m, n, k, lda, alpha);
    auto mem_b_pack = gemm.pack(tensor_b.data(), col_major, false, trans_b, m, n, k, ldb, pack_a ? 1.f : alpha);
    auto status = gemm.execute(pack_a ? mem_a_pack : tensor_a.data(),
                               pack_b ? mem_b_pack :  tensor_b.data(),
                               tensor_c.mutable_data(), tensor_oc.data(),
                               m, n, k, offset_a, offset_b, oc_mode,
                               col_major, trans_a, trans_b, pack_a, pack_b,
                               beta, alpha, lda, ldb, ldc);

    // compute lda, ldb, ldc automatic
    // auto mem_a_pack = gemm.pack(tensor_a.data(), col_major, true, trans_a, m, n, k, alpha);
    // auto mem_b_pack = gemm.pack(tensor_b.data(), col_major, false, trans_b, m, n, k, pack_a ? 1.f : alpha);
    // auto status = gemm.execute(pack_a ? mem_a_pack : tensor_a.data(),
    //                            pack_b ? mem_b_pack : tensor_b.data(),
    //                            tensor_c.mutable_data(), tensor_oc.data(),
    //                            m, n, k,  offset_a, offset_b, oc_mode,
    //                            col_major, trans_a, trans_b, pack_a, pack_b,
    //                            beta, alpha, lda, ldb, ldc);

    gemm.add_offset2mem_c(s8_a, oc_mode, mem_oc_s8a, tensor_c.mutable_data(), m, n);

    gemm.release(mem_oc_s8a);
    gemm.release(mem_a_pack);
    gemm.release(mem_b_pack);
    if (status != S_Success) {
        return S_InvalidValue;
    }

    status = gemm_ref.execute(tensor_a_ref.data(), tensor_b.data(),
                              tensor_oc.data(), tensor_c_ref.mutable_data(),
                              oc_mode, col_major, offset_a, offset_b, m, n, k,
                              trans_a, trans_b, beta, alpha);
    if (status != S_Success) {
        return S_InvalidValue;
    }

    auto count = count_diff<C_DType>(tensor_c.data(), tensor_c_ref.data(),
                                     tensor_c.valid_size(), 5e-3, true, false);

    double quantized_error_rate = 100 * count / tensor_c.valid_size();

    if (quantized_error_rate < 0.5) {
        LOG(INFO) << "Gemm x86 successed, quantized error rate is "
                  << quantized_error_rate << "%\n";
    } else {
        if (a_dtype == DT_FLOAT) {
            LOG(ERROR) << "Gemm x86 {"
                       << " io_dtype:" << io_dtype
                       << " layout:"   << (col_major ? "col" : "row")
                       << " oc_mode:"  << oc_mode
                       << " m:"        << m
                       << " n:"        << n
                       << " k:"        << k
                       << " oa:"       << int32_t(offset_a)
                       << " ob:"       << int32_t(offset_b)
                       << " lda:"      << lda
                       << " ldb:"      << ldb
                       << " ldc:"      << ldc
                       << " beta:"     << beta
                       << " alpha:"    << alpha
                       << " pack_a:"   << (pack_a ? "true" : "false")
                       << " pack_b:"   << (pack_b ? "true" : "false")
                       << " trans_a:"  << (trans_a ? "true" : "false")
                       << " trans_b:"  << (trans_b ? "true" : "false")
                       << " }";
            LOG(ERROR) << "Gemm x86 failed, quantized error rate is "
                       << quantized_error_rate << "%\n";
        }
    }

    return S_Success;
}

TEST(TestFunc, test_inner_product) {

#ifdef USE_X86_PLACE

    #ifdef ICESWORD_DEBUG
        for (int8_t offset_a : {0, 127, -128}) { // float clbas gemm don't support it
        for (int8_t offset_b : {0, 127, -128}) { // float clbas gemm don't support it
        for (auto oc_mode : {'N', 'F', 'R', 'C'}) {
        for (auto col_major : {true}) {
        for (auto trans_a : {false, true}) {
        for (auto trans_b : {false, true}) {
        for (auto pack_a : {false, true}) {
        for (auto pack_b : {false, true}) {
        for (auto alpha : {0.5f}) {
        for (auto beta : {0.f}) {
        for (auto m : {2}) {
        for (auto n : {3}) {
        for (auto k : {100}) {
            // test_gemm_cpu<DT_FLOAT, DT_FLOAT, DT_FLOAT>(m, n, k, offset_a, offset_b, oc_mode, col_major,
            //                                             trans_a, trans_b, pack_a, pack_b, alpha, beta);
            // test_gemm_cpu<DT_UINT8, DT_INT8, DT_INT32>(m, n, k, offset_a, offset_b, oc_mode, col_major,
            //                                            trans_a, trans_b, pack_a, pack_b, alpha, beta);
            test_gemm_cpu<DT_INT8, DT_INT8, DT_INT32>(m, n, k, offset_a, offset_b, oc_mode, col_major,
                                                      trans_a, trans_b, pack_a, pack_b, alpha, beta);

        }}}}}}}}}}}}}
    #else
        for (auto offset_a : {0, 127, -128}) { // float clbas gemm don't support it
        for (auto offset_b : {0, 127, -128}) { // float clbas gemm don't support it
        for (auto oc_mode : {'N', 'F', 'R', 'C'}) {
        for (auto col_major : {false, true}) {
        for (auto trans_a : {false, true}) {
        for (auto trans_b : {false, true}) {
        for (auto pack_a : {false, true}) {
        for (auto pack_b : {false, true}) {
        for (auto alpha : {0.f, 1.f}) {
        for (auto beta : {0.f, 1.f}) {
        for (auto m : {1, 10}) {
        for (auto n : {1, 10}) {
        for (auto k : {1, 10}) {
            test_gemm_cpu<DT_FLOAT, DT_FLOAT, DT_FLOAT>(m, n, k, offset_a, offset_b, oc_mode, col_major,
                                                        trans_a, trans_b, pack_a, pack_b, alpha, beta);
            test_gemm_cpu<DT_UINT8, DT_INT8, DT_INT32>(m, n, k, offset_a, offset_b, oc_mode, col_major,
                                                       trans_a, trans_b, pack_a, pack_b, alpha, beta);
            test_gemm_cpu<DT_UINT8, DT_INT8, DT_INT32>(m, n, k, offset_a, offset_b, oc_mode, col_major,
                                                       trans_a, trans_b, pack_a, pack_b, alpha, beta);
        }}}}}}}}}}}}}
    #endif
#endif

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}