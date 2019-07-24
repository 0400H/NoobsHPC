/* Copyright (c) 2018 ipparam.NoobsHPC Authors All Rights Reserve.

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

#include "bench_common.h"

#define LOOP_WARMUP 50
#define LOOP 200

benchmark_operator_param<AXPY> benchmark_param[] {
    { {LT_NCHW, 50, 1, 100, 100, 1, 3, 200, 3, 3, 1, 1, 1, 1, 1, 1, false, "no_act"}, {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { {LT_NCHW, 50, 1, 100, 100, 1, 3, 200, 3, 3, 1, 1, 1, 1, 1, 1, true, "no_act"}, {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
    { {LT_NCHW, 50, 1, 100, 100, 1, 3, 200, 3, 3, 1, 1, 1, 1, 1, 1, true, "relu"}, {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT} },
};

template <ExecuteMethod AType, DataType OPDType>
Status axpy_init (std::vector<Tensor<X86>*>& input,
                         benchmark_operator_param<AXPY>& param,
                         benchmark_operator_memory<X86, AXPY, AType, OPDType>& memory,
                         bool verbose = false) {
    auto & dtparam = param.dtype_param;
    auto & p = param.conv_param;
    auto oh = (p.ih + 2 * p.ph - p.kh / p.dh) / p.sh + 1;
    auto ow = (p.iw + 2 * p.pw - p.kw / p.dw) / p.sw + 1;
    if (oh <= 0 || ow <= 0) {
        LOG(ERROR) << "Convolution x86 wrong param!\n";
        return S_UnImplError;
    }

    std::vector<int> out_nchw = {p.batch, p.oc, oh, ow},
                     out_nhwc = {p.batch, oh, ow, p.oc};

    // {g, oc/g}
    Shape BiasShape({p.oc}, LT_C);
    // {mb, g, oc/g, oh, ow} or {mb, oh, ow, g, oc/g}
    Shape OutShape(p.layout == LT_NCHW ? out_nchw : out_nhwc, p.layout);
    // {g, oc/g, kh, kw, ic/g}
    Shape WeightShape({p.oc, p.kh, p.kw, p.ic/p.group}, LT_NHWC);

    memory.output.push_back(new Tensor<X86>);

    memory.weight.re_alloc(WeightShape, dtparam.weight_dtype);
    memory.bias.re_alloc(BiasShape, dtparam.bias_dtype);
    memory.output[0]->re_alloc(OutShape, dtparam.output_dtype);

    fill_tensor_rand(memory.weight, -128, 127);
    fill_tensor_rand(memory.bias, -10, 10);

    ImplParam<X86, ACT> act_param(p.act_param.active_type);
    ImplParam<X86, AXPY> impl_param(&memory.weight, p.with_bias ? &memory.bias : nullptr,
                                           p.group, p.sh, p.sw, p.dh, p.dw, p.ph, p.pw, "nearest", act_param);
    memory.param = impl_param;

    if (verbose) {
        auto string_io_layout = get_layout_string(p.layout);
        auto string_active_type = get_algorithm_string(p.act_param.active_type);
        auto string_io_dtype = get_io_dtype_string(dtparam.input_dtype, dtparam.output_dtype);
        LOG(INFO) << "axpy {"
                  << " io_dtype:" << string_io_dtype
                  << " p.layout:" << string_io_layout
                  << " active_type:" << string_active_type
                  << " with_bias:" << (p.with_bias ? "true" : "false")
                  << " mb:" << p.batch
                  << " group:" << p.group
                  << " ic:" << p.ic
                  << " oc:" << p.oc
                  << " ih:" << p.ih
                  << " iw:" << p.iw
                  << " oh:" << oh
                  << " ow:" << ow
                  << " kh:" << p.kh
                  << " kw:" << p.kw
                  << " stride_h:" << p.sh
                  << " stride_w:" << p.sw
                  << " dilation_h:" << p.dh
                  << " dilation_w:" << p.dw
                  << " pad_h:" << p.ph
                  << " pad_w:" << p.pw
                  << " }";
    }

    return memory.op.init(input, memory.output, memory.param);
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);

    for (size_t i = 0; i < ARRAY_SIZE(benchmark_param); i++) {
        LOG(INFO) << "############################## benchmark case " << i << " ##############################";
        auto p = benchmark_param[i].conv_param;
        auto dtparam = benchmark_param[i].dtype_param;

        std::vector<Tensor<X86> *> input;
        std::vector<int> in_nchw = {p.batch, p.ic, p.ih, p.iw},
                         in_nhwc = {p.batch, p.ih, p.iw, p.ic};

        // {mb, g, ic/g, ih, iw} or {mb, ih, iw, g, ic/g}
        Shape InputShape(p.layout == LT_NCHW ? in_nchw : in_nhwc, p.layout);
        input.push_back(new Tensor<X86>);
        input[0]->re_alloc(InputShape, dtparam.input_dtype);
        fill_tensor_rand(*input[0], -100.f, 100.f);

        if (dtparam.weight_dtype == DT_FLOAT) {
            benchmark_operator_memory<X86, AXPY, FWD_GEMM, DT_FLOAT> mem;
            axpy_init<FWD_GEMM, DT_FLOAT>(input, benchmark_param[i], mem, true);
            benchmark_operator_execute<benchmark_operator_memory<X86, AXPY, FWD_GEMM, DT_FLOAT> &>(input, mem, LOOP_WARMUP, false, false);
            benchmark_operator_execute<benchmark_operator_memory<X86, AXPY, FWD_GEMM, DT_FLOAT> &>(input, mem, LOOP, true, true, "Convolution");
        }
    }

    return 0;
}