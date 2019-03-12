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

#define LOOP_WARMUP 500
#define LOOP 10000

benchmark_operator_param<INNERPRODUCT> benchmark_param[] {
    { {1000, 1000, {500}, false, ET_forward_gemm, AT_invalid}, {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { {1000, 1000, {500}, false, ET_forward_gemm, AT_invalid}, {DT_UINT8, DT_INT8, DT_INT32, DT_INT32} },
    { {1000, 1000, {500}, false, ET_forward_gemm, AT_invalid}, {DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT} },
    { {1000, 1000, {500}, false, ET_forward_gemm, AT_invalid}, {DT_UINT8, DT_INT8, DT_INT32, DT_UINT8} },

    { {1000, 1000, {500}, true, ET_forward_gemm, AT_invalid}, {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { {1000, 1000, {500}, true, ET_forward_gemm, AT_invalid}, {DT_UINT8, DT_INT8, DT_INT32, DT_INT32} },
    { {1000, 1000, {500}, true, ET_forward_gemm, AT_invalid}, {DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT} },
    { {1000, 1000, {500}, true, ET_forward_gemm, AT_invalid}, {DT_UINT8, DT_INT8, DT_INT32, DT_UINT8} },

    { {1000, 1000, {500}, true, ET_forward_gemm, AT_relu}, {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { {1000, 1000, {500}, true, ET_forward_gemm, AT_relu}, {DT_UINT8, DT_INT8, DT_INT32, DT_INT32} },
    { {1000, 1000, {500}, true, ET_forward_gemm, AT_relu}, {DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT} },
    { {1000, 1000, {500}, true, ET_forward_gemm, AT_relu}, {DT_UINT8, DT_INT8, DT_INT32, DT_UINT8} },
    };

template <ExecuteMethod AType, DataType OPDType>
Status inner_product_init (std::vector<Tensor<X86>*>& input,
                           benchmark_operator_param<INNERPRODUCT>& param,
                           benchmark_operator_memory<X86, INNERPRODUCT, AType, OPDType>& memory,
                           bool verbose = false) {
    auto & ipparam = param.ip_param;
    auto & dtparam = param.dtype_param;

    int dim_k = 0;
    for (int i = 0; i < ipparam.K.size(); i++) {
        dim_k += ipparam.K[i];
    }

    Shape WeightShape({ipparam.N, dim_k}, LT_NC);
    Shape BiasShape({ipparam.N}, LT_C);
    Shape OutShape({ipparam.M, ipparam.N}, LT_NC);

    memory.output.push_back(new Tensor<X86>);

    memory.weight.re_alloc(WeightShape, dtparam.weight_dtype);
    memory.bias.re_alloc(BiasShape, dtparam.bias_dtype);
    memory.output[0]->re_alloc(OutShape, dtparam.output_dtype);

    fill_tensor_rand(memory.weight, -128, 127);
    fill_tensor_rand(memory.bias, -10, 10);

    if (dtparam.weight_dtype == DT_INT8) {
        for (int i = 0; i < ipparam.N; i ++) {
            memory.weight_scale.push_back(1.f / 256.f);
        }
        memory.weight.set_scale(memory.weight_scale);
        memory.output[0]->set_scale({1.f / 256.f});
    }

    ImplParam<X86, INNERPRODUCT> impl_param(&memory.weight, ipparam.with_bias ? &memory.bias : nullptr,
                                            false, true, ipparam.act_param.active_type); // active todo
    memory.param = impl_param;

    if (verbose) {
        auto string_active_type = get_algorithm_string(ipparam.act_param.active_type);
        auto string_io_dtype = get_io_dtype_string(dtparam.input_dtype, dtparam.output_dtype);
        LOG(INFO)<< "Inner product x86 {"
                 << " io_dtype:" << string_io_dtype
                 << " m:" << ipparam.M
                 << " n:" << ipparam.N
                 << " k:" << dim_k
                 << " bias:" << (ipparam.with_bias ? "true" :"false")
                 << " active:" << string_active_type
                 << " }";
    }

    return memory.op.init(input, memory.output, memory.param);
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);

    for (size_t i = 0; i < ARRAY_SIZE(benchmark_param); i++) {
        LOG(INFO) << "############################## benchmark case " << i << " ##############################";

        auto ipparam = benchmark_param[i].ip_param;
        auto dtparam = benchmark_param[i].dtype_param;

        std::vector<Tensor<X86> *> input;
        for (int i = 0; i < ipparam.K.size(); i++) {
            Shape InputShape({ipparam.M, ipparam.K[i]}, LT_NC);
            input.push_back(new Tensor<X86>);
            input[i]->re_alloc(InputShape, dtparam.input_dtype);
            if (dtparam.input_dtype == DT_FLOAT) {
                fill_tensor_rand(*input[i], -100.f, 100.f);
            } else {
                fill_tensor_rand(*input[i], 0, 255);
                input[i]->set_scale({1.f / 256.f});
            }
        }

        if (dtparam.weight_dtype == DT_FLOAT) {
            benchmark_operator_memory<X86, INNERPRODUCT, ET_forward_gemm, DT_FLOAT> mem;
            inner_product_init<ET_forward_gemm, DT_FLOAT>(input, benchmark_param[i], mem, true);
            benchmark_operator_execute<benchmark_operator_memory<X86, INNERPRODUCT, ET_forward_gemm, DT_FLOAT> &>(input, mem, LOOP_WARMUP, false, false);
            benchmark_operator_execute<benchmark_operator_memory<X86, INNERPRODUCT, ET_forward_gemm, DT_FLOAT> &>(input, mem, LOOP, true, true, "Inner product");
        } else if (dtparam.weight_dtype == DT_INT8) {
            benchmark_operator_memory<X86, INNERPRODUCT, ET_forward_gemm, DT_INT8> mem;
            inner_product_init<ET_forward_gemm, DT_INT8>(input, benchmark_param[i], mem, true);
            benchmark_operator_execute<benchmark_operator_memory<X86, INNERPRODUCT, ET_forward_gemm, DT_INT8> &>(input, mem, LOOP_WARMUP, false, false);
            benchmark_operator_execute<benchmark_operator_memory<X86, INNERPRODUCT, ET_forward_gemm, DT_INT8> &>(input, mem, LOOP, true, true, "Inner product");
        }
    }

    return 0;
}
