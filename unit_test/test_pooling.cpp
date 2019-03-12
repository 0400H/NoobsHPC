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

template <DataType inDtype, DataType outDtype>
Status test_pooling_cpu(int mb, int ic, int ih, int iw,
                        int kh, int kw, int sh, int sw,
                        int ph, int pw, LayoutType layout,
                        AlgorithmType algo_pooling) {
    auto oh = (ih + 2 * ph - kh) / sh + 1;
    auto ow = (iw + 2 * pw - kw) / sw + 1;
    if (oh <= 0 || ow <= 0) {
        LOG(ERROR) << "Convolution x86 wrong param!\n";
        return S_UnImplError;
    }

    auto algo_pool_string = get_algorithm_string(algo_pooling);
    auto layout_string = get_layout_string(layout);
    auto datetype_string = get_io_dtype_string(inDtype, outDtype);
    LOG(INFO) << "Pooling x86 {"
              << " layout:"     << layout_string
              << " dtype:"      << datetype_string
              << " algo:"       << algo_pool_string
              << " kh:"         << kh
              << " kw:"         << kw
              << " sh:"         << sh
              << " sw:"         << sw
              << " ph:"         << ph
              << " pw:"         << pw
              << " }";

    std::vector<Tensor<X86> *> inputs, outputs, outputs_ref;
    Tensor<X86> weights, bias;

    std::vector<int> in_nchw = {mb, ic, ih, iw}, in_nhwc = {mb, ih, iw, ic},
                     out_nchw = {mb, ic, oh, ow}, out_nhwc = {mb, oh, ow, ic};

    Shape InputShape(layout == LT_NCHW ? in_nchw : in_nhwc, layout);
    Shape OutShape(layout == LT_NCHW ? out_nchw : out_nhwc, layout);

    inputs.push_back(new Tensor<X86>);
    outputs.push_back(new Tensor<X86>);
    outputs_ref.push_back(new Tensor<X86>);

    inputs[0]->re_alloc(InputShape, inDtype);
    outputs[0]->re_alloc(OutShape, outDtype);
    outputs_ref[0]->re_alloc(OutShape, outDtype);

    if (inDtype == DT_FLOAT) {
        fill_tensor_rand(*inputs[0], -10.f, 10.f);
        // fill_matrix_debug<DT_FLOAT>(inputs[0]->mutable_data();, mb * g, ic * ih * iw, true, true);
    }

    ImplParam<X86, POOLING> impl_param(kh, kw, sh, sw, ph, pw, algo_pooling);
    Operator<X86, POOLING, ET_forward_gemm, inDtype> pooling_inference;

    auto status = pooling_inference.init(inputs, outputs, impl_param);
    if (status != S_Success) {
        LOG(ERROR) << "Convolution x86 init failed!\n";
        return S_UnImplError;
    }

    pooling_inference.execute(inputs, outputs, impl_param);

    long count = 0;
    if (inDtype == DT_FLOAT && outDtype == DT_FLOAT) {
        // pooling_reference<float, float, float, float, X86>(inputs, outputs_ref, impl_param);
        // count = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
        //                           outputs[0]->valid_size(), 1e-3, false);
    }

    double quantized_error_rate = 100 * count / outputs[0]->valid_size();

    if (quantized_error_rate < 0.05) {
        LOG(INFO) << "Test pooling x86 successed, quantized error is "
                  << quantized_error_rate << "%\n";
    } else {
        LOG(ERROR) << "Pooling x86 {"
                   << " layout:"     << layout_string
                   << " dtype:"      << datetype_string
                   << " algo:"       << algo_pool_string
                   << " kh:"         << kh
                   << " kw:"         << kw
                   << " sh:"         << sh
                   << " sw:"         << sw
                   << " ph:"         << ph
                   << " pw:"         << pw
                   << " }";
        LOG(ERROR) << "Test pooling x86 failed, quantized error is "
                   << quantized_error_rate << "%\n";
    }

    return S_Success;
}

TEST(TestFunc, test_pooling) {

#ifdef USE_X86_PLACE

    #ifdef ICESWORD_DEBUG
        for (auto layout : {LT_NCHW}) {
        for (auto algo : {AT_max, AT_mean}) {
        for (auto mb : {5}) {
        for (auto ic : {3}) {
        for (auto ih : {3}) {
        for (auto iw : {3}) {
        for (auto kh : {3}) {
        for (auto kw : {3}) {
        for (auto sh : {1}) {
        for (auto sw : {1}) {
        for (auto ph : {1}) {
        for (auto pw : {1}) {
            test_pooling_cpu<DT_FLOAT, DT_FLOAT>(mb, ic, ih, iw, kh, kw, sh, sw, ph, pw, layout, algo);
        }}}}}}}}}}}}
    #else

    #endif

#endif

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}