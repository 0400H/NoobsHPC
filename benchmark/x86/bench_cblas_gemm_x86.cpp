/* Copyright (c) 2018 ipparam.NoobsDNN Authors All Rights Reserve.

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

#include "bench_common_x86.h"

// #define ICESWORD_VERBOSE
#define LOOP_WARMUP 20
#define LOOP 100

typedef struct {
    size_t M;
    size_t N;
    size_t K;
    float beta;
    float alpha;
    bool trans_a;
    bool trans_b;
    char offset_mode;
    bench_datatype_param dtparam;
} bench_cblas_gemm_param;

bench_cblas_gemm_param benchmark_param[] {
    { 5000, 5000, 1000, 0.f, 1.f, false, false, 'N', {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { 5000, 5000, 1000, 0.f, 1.f, false, true, 'N', {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { 5000, 5000, 1000, 0.f, 1.f, false, true, 'C', {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },

    { 5000, 5000, 1000, 0.f, 1.f, false, false, 'N', {DT_UINT8, DT_INT8, DT_INT32, DT_INT32} },
    { 5000, 5000, 1000, 0.f, 1.f, false, true, 'N', {DT_UINT8, DT_INT8, DT_INT32, DT_INT32} },
    { 5000, 5000, 1000, 0.f, 1.f, false, true, 'C', {DT_UINT8, DT_INT8, DT_INT32, DT_INT32} },
};

template <DataType opDtype>
Status benchmark_gemm_cpu(bench_cblas_gemm_param& param,
                          bool verbose = false) {
    auto dtparam = param.dtparam;
    auto M = param.M;
    auto N = param.N;
    auto K = param.K;
    auto alpha = param.alpha;
    auto beta = param.beta;
    auto trans_a = param.trans_a;
    auto trans_b = param.trans_b;
    auto offset_mode = param.offset_mode;

    if (verbose) {
        auto io_dtype = get_io_dtype_string(dtparam.input_dtype, dtparam.output_dtype);
        LOG(INFO)<< "Cblas gemm x86 {"
                << " io_dtype: " << io_dtype
                << " offset_mode:" << offset_mode
                << " trans_a: " << (trans_a ? "true" : "false")
                << " trans_b: " << (trans_b ? "true" : "false")
                << " m: " << M
                << " n: " << N
                << " k: " << K
                << " }";
    }

    class Timer timer;
    CBLAS_GEMM<X86, opDtype> gemm;
    Tensor<X86> input, weight, offset, output, output_ref;

    Shape InputShape({M, K}, LT_NC);
    Shape WeightShape({N, K}, LT_NC);
    Shape OffsetShape({N}, LT_C);
    Shape OutShape({M, N}, LT_NC);

    weight.re_alloc(WeightShape, opDtype);
    input.re_alloc(InputShape, dtparam.input_dtype);
    offset.re_alloc(OffsetShape, dtparam.bias_dtype);
    output.re_alloc(OutShape, dtparam.output_dtype);
    output_ref.re_alloc(OutShape, dtparam.output_dtype);

    fill_tensor_rand(input, 0, 255);
    fill_tensor_rand(weight, -128, 127);
    fill_tensor_rand(offset, -100, 100);

    for (int i = 0; i < LOOP_WARMUP; i++) {
        gemm.execute(input.data(), weight.data(), output.mutable_data(), offset.data(),
                     M, N, K, 0, 0, offset_mode, true, trans_a, trans_b, false, false, beta, alpha);
    }

    for (int i = 0; i < LOOP; i++) {
        timer.start();
        gemm.execute(input.data(), weight.data(), output.mutable_data(), offset.data(),
                     M, N, K, 0, 0, offset_mode, true, trans_a, trans_b, false, false, beta, alpha);
        timer.stop();
    }
    benchmark_timer(timer, "cblas gemm x86");

    return S_Success;
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);

    for (size_t i = 0; i < ARRAY_SIZE(benchmark_param); i++) {
        LOG(INFO) << "############################## benchmark case " << i << " ##############################";
        auto wei_dtype = benchmark_param[i].dtparam.weight_dtype;
        if (wei_dtype == DT_FLOAT) {
            benchmark_gemm_cpu<DT_FLOAT>(benchmark_param[i]);
        } else if (wei_dtype == DT_INT8) {
            benchmark_gemm_cpu<DT_INT8>(benchmark_param[i]);
        }
    }

    return 0;
}