/* Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

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

#include "test_common.h"

template <DataType ADtype,
          DataType BDtype,
          DataType BiasDtype,
          DataType CDtype>
Status test_ip_cpu(const int M,
                   const int N,
                   std::vector<int> K,
                   const bool trans_a = false,
                   const bool trans_b = false,
                   const bool with_bias = false,
                   const bool with_active = false) {
    int total_k = 0;
    for (int i = 0; i < K.size(); i++) {
        total_k += K[i];
    }

    auto io_dtype = get_io_dtype_string(ADtype, CDtype);
    LOG(INFO)<< "Inner product{"
             << " io_dtype:" << io_dtype
             << " m:" << M
             << " n:" << N
             << " k:" << total_k
             << " trans_a:" << (trans_a ? "true" : "false")
             << " trans_b:" << (trans_b ? "true" : "false")
             << " with_bias:" << (with_bias ? "true" : "false")
             << " with_active:" << (with_active ? "relu" : "false")
             << " }";

    std::vector<Tensor<X86> *> matrix_a, matrix_c, matrix_c_ref;
    std::vector<float> scale_matrix_b;
    Tensor<X86> matrix_b, bias;

    Shape BiasShape({N}, LT_C);
    Shape OutShape({M, N}, LT_NC);
    Shape MatrixBShape({N, total_k}, LT_NC);  // {N * dim_k1, ... ,N * dim_kn}

    matrix_c.push_back(new Tensor<X86>);
    matrix_c_ref.push_back(new Tensor<X86>);

    for (int i = 0; i < K.size(); i++) {
        Shape MatrixAShape({M, K[i]}, LT_NC);
        matrix_a.push_back(new Tensor<X86>);
        matrix_a[i]->re_alloc(MatrixAShape, ADtype);
        if (ADtype == DT_FLOAT) {
            fill_tensor_rand(*matrix_a[i], -100.f, 100.f);
        } else {
            fill_tensor_rand(*matrix_a[i], 0, 255);
            matrix_a[i]->set_scale({1.f / 2.f});
        }
    }

    matrix_b.re_alloc(MatrixBShape, BDtype);
    bias.re_alloc(BiasShape, BiasDtype);
    matrix_c[0]->re_alloc(OutShape, CDtype);
    matrix_c_ref[0]->re_alloc(OutShape, CDtype);

    fill_tensor_rand(matrix_b, -128, 127);
    fill_tensor_rand(bias, -100, 100);

    if (ADtype == DT_UINT8) {
        for (int i = 0; i < N; i ++) {
            scale_matrix_b.push_back(1.f / 2.f);
        }
        matrix_b.set_scale(scale_matrix_b);
        matrix_c[0]->set_scale({1.f / 2.f});
        matrix_c_ref[0]->set_scale({1.f / 2.f});
    }

    ImplParam<X86, INNERPRODUCT> impl_param(&matrix_b, with_bias ? &bias : nullptr,
                                            trans_a, trans_b, with_active);
    Operator<X86, INNERPRODUCT, ET_forward_gemm, BDtype> ip_inference;

    auto status = ip_inference.init(matrix_a, matrix_c, impl_param);
    if (status != S_Success) {
        LOG(ERROR) << "Inner product x86 init failed!\n";
        return S_UnImplError;
    }

    ip_inference.execute(matrix_a, matrix_c, impl_param);

    long count = 0;
    if (ADtype == DT_FLOAT && CDtype == DT_FLOAT) {
        ip_reference<float, float, float, float, X86>(matrix_a, matrix_c_ref, impl_param);
        count = count_diff<float>(matrix_c[0]->data(), matrix_c_ref[0]->data(),
                                  matrix_c[0]->valid_size(), 1e-3,  true, false);
    } else if (ADtype == DT_UINT8 && CDtype == DT_INT32){
        ip_reference<uint8_t, int8_t, int32_t, int32_t, X86>(matrix_a, matrix_c_ref, impl_param);
        count = count_diff<int32_t>(matrix_c[0]->data(), matrix_c_ref[0]->data(),
                                    matrix_c[0]->valid_size(), 1e-5,  true, false);
    } else if (ADtype == DT_UINT8 && CDtype == DT_FLOAT) {
        ip_reference<uint8_t, int8_t, int32_t, float, X86>(matrix_a, matrix_c_ref, impl_param);
        count = count_diff<float>(matrix_c[0]->data(), matrix_c_ref[0]->data(),
                                  matrix_c[0]->valid_size(), 1e-5,  true, false);
    } else if (ADtype == DT_UINT8 && CDtype == DT_UINT8){
        ip_reference<uint8_t, int8_t, int32_t, uint8_t, X86>(matrix_a, matrix_c_ref, impl_param);
        count = count_diff<uint8_t>(matrix_c[0]->data(), matrix_c_ref[0]->data(),
                                    matrix_c[0]->valid_size(), 1e-5,  true, false);
    }

    double quantized_error_rate = 100 * count / matrix_c[0]->valid_size();

    if (quantized_error_rate < 0.5) {
        LOG(INFO) << "Inner product x86 successed, quantized error rate is "
                  << quantized_error_rate << "%\n";
    } else {
        LOG(ERROR)<< "Inner product{"
                  << " io_dtype:" << io_dtype
                  << " m:" << M
                  << " n:" << N
                  << " k:" << total_k
                  << " trans_a:" << (trans_a ? "true" : "false")
                  << " trans_b:" << (trans_b ? "true" : "false")
                  << " with_bias:" << (with_bias ? "true" : "false")
                  << " with_active:" << (with_active ? "true" : "false")
                  << " }";
        LOG(ERROR) << "Inner product x86 failed, quantized error rate is "
                   << quantized_error_rate << "%\n";
    }

    return S_Success;
}

TEST(TestFunc, test_inner_product) {

#ifdef USE_X86_PLACE
    #ifdef ICESWORD_DEBUG
        for (auto trans_a : {false}) {
        for (auto trans_b : {true}) {
        for (auto with_bias : {false, true}) {
        for (auto with_active : {false, true}) {
        for (auto m : {100}) {
        for (auto n : {200}) {
        for (auto k : {std::vector<int>{1, 50, 400}}) {
            test_ip_cpu<DT_FLOAT,
                        DT_FLOAT,
                        DT_FLOAT,
                        DT_FLOAT>(m, n, k, trans_a, trans_b, with_bias, with_active);
        }}}}}}}
    #else
        for (auto trans_a : {false}) {
        for (auto trans_b : {true}) {
        for (auto with_bias : {false, true}) {
        for (auto with_active : {false, true}) {
        for (auto m : {1, 100, 1000}) {
        for (auto n : {1, 100, 1000}) {
        for (auto k : {std::vector<int>{500},
                       std::vector<int>{1, 500, 1000}}) {
            test_ip_cpu<DT_FLOAT,
                        DT_FLOAT,
                        DT_FLOAT,
                        DT_FLOAT>(m, n, k, trans_a, trans_b, with_bias, with_active);
            test_ip_cpu<DT_UINT8,
                        DT_INT8,
                        DT_INT32,
                        DT_INT32>(m, n, k, trans_a, trans_b, with_bias, with_active);
            test_ip_cpu<DT_UINT8,
                        DT_INT8,
                        DT_INT32,
                        DT_FLOAT>(m, n, k, trans_a, trans_b, with_bias, with_active);
            test_ip_cpu<DT_UINT8,
                        DT_INT8,
                        DT_INT32,
                        DT_UINT8>(m, n, k, trans_a, trans_b, with_bias, with_active);
        }}}}}}}
    #endif

#endif

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}