/* Copyright (c) 2018 NoobsHPC Authors All Rights Reserve.

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
// #define ICESWORD_DEBUG

template <DataType inDtype, DataType outDtype>
Status test_axpy_cpu(int mb, int ic, bool with_bias) {
    auto io_dtype = get_io_dtype_string(inDtype, outDtype);
    LOG(INFO) << "Axpy x86 {"
              << " dtype:"      << io_dtype
              << " batch:"      << mb
              << " ic:"         << ic
              << " with_bias:"  << (with_bias ? "true" : "false")
              << " }";

    Shape Shape_C({ic}, LT_C);
    Shape Shape_NC({mb, ic}, LT_NC);

    Tensor<X86> alpha, input, bias, output, output_ref;
    alpha.re_alloc(Shape_C, DT_FLOAT);
    bias.re_alloc(Shape_C, outDtype);
    input.re_alloc(Shape_NC, inDtype);
    output.re_alloc(Shape_NC, outDtype);
    output_ref.re_alloc(Shape_NC, outDtype);
    output_ref.copy_from(output);

    std::vector<Tensor<X86> *> inputs, outputs, outputs_ref;
    inputs.push_back(&input);
    outputs.push_back(&output);
    outputs_ref.push_back(&output_ref);

    #ifdef ICESWORD_DEBUG
        if (inDtype == DT_FLOAT) {
            fill_tensor_debug<DT_FLOAT>(input.mutable_data(), mb, ic, true, true);
            fill_tensor_const(bias, 10.f);
            fill_tensor_const(alpha, 10.f);
        }
    #else
        if (inDtype == DT_FLOAT) {
            fill_tensor_rand(input, -10.f, 10.f);
            fill_tensor_rand(bias, -10, 10);
            fill_tensor_rand(alpha, -10.f, 10.f);
        }
    #endif

    ImplParam<X86, AXPY> impl_param(&alpha, with_bias ? &bias : nullptr);
    Operator<X86, AXPY, FWD_REF, inDtype> axpy_reference;
    // Operator<X86, AXPY, FWD_SSE, inDtype> axpy_inference;
    Operator<X86, AXPY, FWD_AVX2, inDtype> axpy_inference;
    // Operator<X86, AXPY, FWD_AVX2_JIT, inDtype> axpy_inference;

    auto inference_status = axpy_inference.init(inputs, outputs, impl_param);
    auto reference_status = axpy_reference.init(inputs, outputs_ref, impl_param);
    if (inference_status != S_Success || reference_status != S_Success) {
        LOG(ERROR) << "Axpy x86 init failed!\n";
        return S_UnImplError;
    }

    axpy_inference.execute(inputs, outputs, impl_param);
    axpy_reference.execute(inputs, outputs_ref, impl_param);

    long count = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
                                   outputs[0]->valid_size(), 1e-3, true, false);

    double quantized_error_rate = 100.0 * count / outputs[0]->valid_size();

    if (quantized_error_rate < 0.05) {
        LOG(INFO) << "Test axpy x86 successed, quantized error is "
                  << quantized_error_rate << "%\n";
    } else {
        LOG(ERROR) << "Axpy x86 {"
                   << " dtype:"       << io_dtype
                   << " batch:"       << mb
                   << " ic:"          << ic
                   << " with_bias:"   << (with_bias ? "true" : "false")
                   << " }";
        LOG(ERROR) << "Test axpy x86 failed, quantized error is "
                   << quantized_error_rate << "%\n";
    }

    return S_Success;
}

TEST(TestFunc, test_convolution) {

#ifdef USE_X86_PLACE
    #ifdef ICESWORD_DEBUG
        for (auto mb : {2}) {
        for (auto ic : {1}) {
        for (auto with_bias : {false, true}) {
            test_axpy_cpu<DT_FLOAT, DT_FLOAT>(mb, ic, with_bias);
        }}}
    #else
        for (auto mb : {1, 5}) {
        for (auto ic : {1, 12, 16, 20}) {
        for (auto with_bias : {false, true}) {
            test_axpy_cpu<DT_FLOAT, DT_FLOAT>(mb, ic, with_bias);
        }}}
    #endif

#endif

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}