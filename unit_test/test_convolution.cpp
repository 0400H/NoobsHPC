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

template <DataType inDtype, DataType opDtype,
          DataType biasDtype, DataType outDtype>
Status test_conv_cpu(int mb, int g, int ih, int iw,
                     int ic, int oc, int kh, int kw,
                     int sh, int sw, int ph, int pw,
                     int dh, int dw, bool with_bias,
                     AlgorithmType algo_act,
                     LayoutType layout = LT_NCHW) {
    auto oh = (ih + 2 * ph - kh / dh) / sh + 1;
    auto ow = (iw + 2 * pw - kw / dw) / sw + 1;
    if (oh <= 0 || ow <= 0) {
        LOG(ERROR) << "Convolution x86 wrong param!\n";
        return S_UnImplError;
    }

    auto act_type = get_algorithm_string(algo_act);
    auto io_layout = get_layout_string(layout);
    auto io_dtype = get_io_dtype_string(inDtype, outDtype);
    LOG(INFO) << "Convolution x86 {"
              << " dtype:"      << io_dtype
              << " layout:"     << io_layout
              << " act_type:"   << act_type
              << " with_bias:"  << (with_bias ? "true" : "false")
              << " batch:"      << mb
              << " group:"      << g
              << " ic:"         << ic
              << " oc:"         << oc
              << " ih:"         << ih
              << " iw:"         << iw
              << " oh:"         << oh
              << " ow:"         << ow
              << " kh:"         << kh
              << " kw:"         << kw
              << " stride_h:"   << sh
              << " stride_w:"   << sw
              << " dilation_h:" << dh
              << " dilation_w:" << dw
              << " pad_h:"      << ph
              << " pad_w:"      << pw
              << " }";

    std::vector<Tensor<X86> *> inputs, outputs, outputs_ref;
    Tensor<X86> weights, bias;

    std::vector<int> in_nchw = {mb, ic, ih, iw}, in_nhwc = {mb, ih, iw, ic},
                     out_nchw = {mb, oc, oh, ow}, out_nhwc = {mb, oh, ow, oc};

    // {g, oc/g}
    Shape BiasShape({oc}, LT_C);
    // {mb, g, ic/g, ih, iw} or {mb, ih, iw, g, ic/g}
    Shape InputShape(layout == LT_NCHW ? in_nchw : in_nhwc, layout);
    // {mb, g, oc/g, oh, ow} or {mb, oh, ow, g, oc/g}
    Shape OutShape(layout == LT_NCHW ? out_nchw : out_nhwc, layout);
    Shape WeightShape({g, oc/g, kh, kw, ic/g}, LT_GOHWI);

    inputs.push_back(new Tensor<X86>);
    outputs.push_back(new Tensor<X86>);
    outputs_ref.push_back(new Tensor<X86>);

    bias.re_alloc(BiasShape, biasDtype);
    weights.re_alloc(WeightShape, opDtype);
    inputs[0]->re_alloc(InputShape, inDtype);
    outputs[0]->re_alloc(OutShape, outDtype);
    outputs_ref[0]->re_alloc(OutShape, outDtype);

    #ifdef ICESWORD_DEBUG
        if (inDtype == DT_FLOAT) {
            fill_matrix_debug<DT_FLOAT>(inputs[0]->mutable_data(), mb * g, ic/g * ih * iw, true, true);
            fill_tensor_const(weights, 1);
            fill_tensor_const(bias, 10);
        }
    #else
        if (inDtype == DT_FLOAT) {
            fill_tensor_rand(*inputs[0], -10.f, 10.f);
            fill_tensor_rand(weights, -128, 127);
            fill_tensor_rand(bias, -10, 10);
        }
    #endif

    ImplParam<X86, ACTIVATION> act_param(algo_act);
    ImplParam<X86, CONVOLUTION> impl_param(&weights, with_bias ? &bias : nullptr,
                                           mb, g, ih, iw, ic, oh, ow, oc, kh, kw,
                                           sh, sw, dh, dw, ph, pw, AT_nearest, act_param);
    Operator<X86, CONVOLUTION, ET_forward_gemm, opDtype> conv_inference;

    auto status = conv_inference.init(inputs, outputs, impl_param);
    if (status != S_Success) {
        LOG(ERROR) << "Convolution x86 init failed!\n";
        return S_UnImplError;
    }

    conv_inference.execute(inputs, outputs, impl_param);

    long count = 0;
    if (inDtype == DT_FLOAT && outDtype == DT_FLOAT) {
        conv_reference<float, float, float, float, X86>(inputs, outputs_ref, impl_param);
        count = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
                                  outputs[0]->valid_size(), 1e-3,  true, false);
    }

    double quantized_error_rate = 100 * count / outputs[0]->valid_size();

    if (quantized_error_rate < 0.05) {
        LOG(INFO) << "Test convolution x86 successed, quantized error is "
                  << quantized_error_rate << "%\n";
    } else {
        LOG(ERROR) << "Convolution x86 {"
                   << " dtype:"       << io_dtype
                   << " layout:"      << io_layout
                   << " act_type:"    << act_type
                   << " with_bias:"   << (with_bias ? "true" : "false")
                   << " batch:"       << mb
                   << " group:"       << g
                   << " ic:"          << ic
                   << " oc:"          << oc
                   << " ih:"          << ih
                   << " iw:"          << iw
                   << " oh:"          << oh
                   << " ow:"          << ow
                   << " kh:"          << kh
                   << " kw:"          << kw
                   << " pad_h:"       << ph
                   << " pad_w:"       << pw
                   << " stride_h:"    << sh
                   << " stride_w:"    << sw
                   << " dilation_h:"  << dh
                   << " dilation_w:"  << dw
                   << " }";
        LOG(ERROR) << "Test convolution x86 failed, quantized error is "
                   << quantized_error_rate << "%\n";
    }



    return S_Success;
}

TEST(TestFunc, test_convolution) {

#ifdef USE_X86_PLACE

    #ifdef ICESWORD_DEBUG
        for (auto layout : {LT_NCHW}) {
        for (auto algo_act : {AT_invalid, AT_relu}) {
        for (auto with_bias : {false, true}) {
        for (auto mb : {5}) {
        for (auto g  : {1, 3}) {
        for (auto ih : {3}) {
        for (auto iw : {3}) {
        for (auto ic : {3}) {
        for (auto oc : {3}) {
        for (auto kh : {3}) {
        for (auto kw : {3}) {
        for (auto sh : {1}) {
        for (auto sw : {1}) {
        for (auto dh : {1}) {
        for (auto dw : {1}) {
        for (auto ph : {1}) {
        for (auto pw : {1}) {
            test_conv_cpu<DT_FLOAT, DT_FLOAT,
                          DT_FLOAT, DT_FLOAT>(mb, g, ih, iw, ic, oc, kh, kw, sh, sw,
                                              ph, pw, dh, dw, with_bias, algo_act, layout);
        }}}}}}}}}}}}}}}}}
    #else
        for (auto layout : {LT_NCHW}) {
        for (auto algo_act : {AT_invalid, AT_relu}) {
        for (auto with_bias : {false, true}) {
        for (auto mb : {1, 5}) {
        for (auto g  : {1, 6}) {
        for (auto ih : {3, 14}) {
        for (auto iw : {3, 14}) {
        for (auto ic : {1, 3, 6}) {
        for (auto oc : {1, 3, 6}) {
        for (auto kh : {1, 3, 5}) {
        for (auto kw : {1, 3, 5}) {
        for (auto sh : {1, 2}) {
        for (auto sw : {1, 2}) {
        for (auto dh : {1, 2}) {
        for (auto dw : {1, 2}) {
        for (auto ph : {0, 1, 2}) {
        for (auto pw : {0, 1, 2}) {
            test_conv_cpu<DT_FLOAT, DT_FLOAT,
                          DT_FLOAT, DT_FLOAT>(mb, g, ih, iw, ic, oc, kh, kw, sh, sw,
                                              ph, pw, dh, dw, with_bias, algo_act, layout);
        }}}}}}}}}}}}}}}}}
    #endif

#endif

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}