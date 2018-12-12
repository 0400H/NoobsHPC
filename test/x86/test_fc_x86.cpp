#include "test_common.h"
#include "icesword/layer/x86/fc_x86.h"

using namespace noobsdnn::icesword;

template <typename dtype>
int count_diff(const void *input1, const void *input2, int size,
               double max_ratio ,bool with_print = false) {
    auto src1 = static_cast<const dtype*>(input1);
    auto src2 = static_cast<const dtype*>(input2);

    if (max_ratio <= 0) {
        max_ratio = 1e-2;
    }

    int count = 0;
    for (int i = 0; i < size; ++i) {
        double ratio = fabs(src1[i] - src2[i]) /
                       fabs(src1[i] + src2[i] + 1e-12);
        if (ratio > max_ratio ) {
            if (with_print) {
                LOG(ERROR) << "out = "<< (float)src1[i]
                           << "\nout_ref = " << (float)src2[i];
            }
            ++count;
        }
    }
    return count;
}

template <typename src_dtype,
          typename op_dtype,
          typename dst_dtype,
          typename bias_dtype,
          TargetType TType>
void fc_cpu_common(const std::vector<Tensor<TType>* > &src,
                   std::vector<Tensor<TType>* > &dst,
                   Param<TType, FC> &param) {
    int output_channel = dst[0]->count_valid(1, dst[0]->dims());
    int batch_size = src[0]->num();

    Shape OutShape({batch_size, output_channel, 1, 1}, LT_NCHW);
    Tensor<X86> dst_tmp;
    dst_tmp.re_alloc(OutShape, DT_INT32);

    auto dst_tmp_data = static_cast<int32_t *>(dst_tmp.mutable_data());
    auto dst_data = static_cast<dst_dtype *>(dst[0]->mutable_data());
    auto weights_data = static_cast<const op_dtype *>(param.weights->data());
    auto bias_data = param.bias ?
                     static_cast<const bias_dtype *>(param.bias->data()):
                     nullptr;

    for (int i = 0; i < src.size(); i++) {
        int IC = src[i]->count_valid(1, src[i]->dims());
        auto src_data = static_cast<const src_dtype *>(src[i]->data());

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mb = 0; mb < batch_size; mb++) {
            for (int oc = 0; oc < output_channel; oc++) {
                int oidx = mb * output_channel + oc;
                if (i == 0) {
                    if (src[0]->get_dtype() == DT_UINT8) {
                        dst_tmp_data[oidx] = bias_data ? bias_data[oc] : dst_dtype{0};
                    } else {
                        dst_data[oidx] = bias_data ? bias_data[oc] : dst_dtype{0};
                    }
                }
                for (int ic = 0; ic < IC; ic++) {
                    int iidx = mb * IC + ic;
                    int widx = oc * IC + ic;
                    if (src[0]->get_dtype() == DT_UINT8) {
                        dst_tmp_data[oidx] += src_data[iidx] * weights_data[widx];
                    } else {
                        dst_data[oidx] += src_data[iidx] * weights_data[widx];
                    }
                }
            }
        }
        weights_data += output_channel * IC;
    }

    if (src[0]->get_dtype() == DT_UINT8) {
        for (int mb = 0; mb < batch_size; mb++) {
            for (int oc = 0; oc < output_channel; oc++) {
                int dst_index = mb * output_channel + oc;
                float scale = (src[0]->get_scale()[0] * param.weights->get_scale()[oc]) /
                              dst[0]->get_scale()[0];
                dst_data[dst_index] = scale * dst_tmp_data[dst_index];
            }
        }
    }
}

template <DataType inDtype,
          DataType opDtype,
          DataType outDtype,
          DataType biasDtype>
void test_fc_cpu(int mb,
                 std::vector<int> ic,
                 int oc,
                 bool with_bias = false,
                 std::vector<float>scale = {1.f, 1.f, 1.f},
                 LayoutType layout = LT_NCHW) {
    std::vector<Tensor<X86> *> inputs, outputs, outputs_ref;
    Tensor<X86> weights, bias;

    int total_ic = 0;
    for (int i = 0; i < ic.size(); i++) {
        total_ic += ic[i];
        Shape InputShape({mb, layout == LT_NCHW ? ic[i] : 1,
                         1, layout == LT_NCHW ? 1 : ic[i]}, layout);
        inputs.push_back(new Tensor<X86>);
        inputs[i]->re_alloc(InputShape, inDtype);
        if (inDtype == DT_FLOAT) {
            fill_tensor_rand(*inputs[i], -10.f, 10.f);
        } else {
            fill_tensor_rand(*inputs[i], 0, 255);
            inputs[i]->set_scale({scale[0]});
        }
    }

    Shape WeightShape({oc, layout == LT_NCHW ? total_ic : 1,
                      1, layout == LT_NCHW ? 1 : total_ic}, layout);
    Shape BiasShape({layout == LT_NCHW ? oc : 1, 1,
                    1, layout == LT_NCHW ? 1 :oc}, layout);
    Shape OutShape({mb, layout == LT_NCHW ? oc : 1,
                   1, layout == LT_NCHW ? 1 : oc}, layout);

    outputs.push_back(new Tensor<X86>);
    outputs_ref.push_back(new Tensor<X86>);

    weights.re_alloc(WeightShape, opDtype);
    bias.re_alloc(BiasShape, biasDtype);
    outputs[0]->re_alloc(OutShape, outDtype);
    outputs_ref[0]->re_alloc(OutShape, outDtype);

    fill_tensor_rand(weights, -10, 10);
    fill_tensor_rand(bias, -10, 10);

    std::vector<float> scale_weights;
    for (int i = 0; i < oc; i ++) {
        scale_weights.push_back(scale[1]);
    }
    weights.set_scale(scale_weights);
    outputs[0]->set_scale({scale[2]});
    outputs_ref[0]->set_scale({scale[2]});

    Param<X86, FC> param(&weights, with_bias ? &bias : nullptr, oc, FORWARD_FC_GEMM);
    Layer<X86, FC, opDtype> Fc_inference;

    Fc_inference.init(inputs, outputs, param);
    Fc_inference.run(inputs, outputs, param);

    int flag = 10;
    if (opDtype == DT_FLOAT) {
        fc_cpu_common<float, float, float, float, X86>(inputs, outputs_ref, param);
        flag = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
                                 outputs[0]->valid_size(), 1e-3);
    } else {
        if (outDtype == DT_FLOAT) {
            fc_cpu_common<uint8_t, int8_t, float, int32_t, X86>(inputs, outputs_ref, param);
            flag = count_diff<float>(outputs[0]->data(), outputs_ref[0]->data(),
                                     outputs[0]->valid_size(), 1e-5);
        } else if (outDtype == DT_INT32){
            fc_cpu_common<uint8_t, int8_t, int32_t, int32_t, X86>(inputs, outputs_ref, param);
            flag = count_diff<int32_t>(outputs[0]->data(), outputs_ref[0]->data(),
                                       outputs[0]->valid_size(), 1e-5);
        } else if (outDtype == DT_INT8){
            fc_cpu_common<uint8_t, int8_t, int8_t, int32_t, X86>(inputs, outputs_ref, param);
            flag = count_diff<int8_t>(outputs[0]->data(), outputs_ref[0]->data(),
                                      outputs[0]->valid_size(), 1e-5);
        }
    }

    if (flag <= 5) {
        LOG(INFO) << "Test fc x86 passed";
    } else {
        LOG(ERROR) << "Test fc x86 failed";
    }

    return;
}

TEST(TestNBFunc, test_op_fc) {

#ifdef USE_X86_PLACE
    for (auto scale : {std::vector<float>{1.f, 1.f ,1.f},
                       std::vector<float>{1.1e-1, 1.2e-1, 1.5e-2}}) {
        for (bool with_bias : {false, true}) {
            for (int w_in : {28}) {
                for (int h_in : {28}) {
                    for (int ch_in : {64}) {
                        for (int num_in : {1, 32}) {
                            for (int out_num : {64, 128}) {
                                test_fc_cpu<DT_FLOAT,
                                            DT_FLOAT,
                                            DT_FLOAT,
                                            DT_FLOAT>(num_in, {ch_in * h_in * w_in}, out_num, with_bias);
                                test_fc_cpu<DT_UINT8,
                                            DT_INT8,
                                            DT_FLOAT,
                                            DT_INT32>(num_in, {ch_in * h_in * w_in}, out_num, with_bias, scale);
                                test_fc_cpu<DT_UINT8,
                                            DT_INT8,
                                            DT_INT32,
                                            DT_INT32>(num_in, {ch_in * h_in * w_in}, out_num, with_bias, scale);
                                test_fc_cpu<DT_UINT8,
                                            DT_INT8,
                                            DT_INT8,
                                            DT_INT32>(num_in, {ch_in * h_in * w_in}, out_num, with_bias, scale);
                            }
                        }
                    }
                }
            }
        }
    }
#endif

}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}