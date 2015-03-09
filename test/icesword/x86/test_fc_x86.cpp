#include "test_fc_x86.h"


using namespace noobsdnn::icesword;

struct test_inner_product_descr_t {
    int mb;
    vector<int> ic;
    int oc;
    int kh;
    int kw;
};

struct inprod_test_params {
    test_inner_product_descr_t test_ipd;
    bool with_bias;
    float scale;
};

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
void compute_ref_inner_product_fwd(vector<Tensor<X86, inDtype, LayOutType_in>* > &src,
                                   Tensor<X86, outDtype, LayOutType_out> &dst,
                                   FcParam<Tensor<X86, OpDtype, LayOutType_op>> &param) {
    typedef Tensor<X86, inDtype, LayOutType_in> inTensor;
    typedef Tensor<X86, OpDtype, LayOutType_op> opTensor;
    typedef Tensor<X86, outDtype, LayOutType_out> outTensor;
    typedef typename inTensor::Dtype src_data_t;
    typedef typename opTensor::Dtype weight_data_t;
    typedef typename outTensor::Dtype dst_data_t;

    dst_data_t *dst_data = static_cast<dst_data_t*>(dst.mutable_data());
    int OC = dst.count_valid(1, dst.dims());
    const weight_data_t *weights_data = static_cast<const weight_data_t*>(param.weights->data());
    const float *bias_data_fp32 = NULL;
    const int *bias_data_s32 = NULL;
    if (param.bias_s32) {
        bias_data_s32 = static_cast<const int *>(param.bias_s32->data());
    }
    if (param.bias_fp32) {
        bias_data_fp32 = static_cast<const float *>(param.bias_fp32->data());
    }
    for (size_t i = 0; i < src.size(); i++) {
        const src_data_t *src_data = static_cast<const src_data_t*>(src[i]->data());
        int IC = src[i]->count_valid(1, src[i]->dims());

        #pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < src[i]->num(); n++) {
            for (int oc = 0; oc < OC; oc++) {
                int oidx = n * OC + oc;
                if (i == 0) {
                    if (param.bias_s32) {
                        dst_data[oidx] = param.scale * bias_data_s32[oc];
                    } else if (param.bias_fp32) {
                        dst_data[oidx] = bias_data_fp32[oc];
                    } else {
                        dst_data[oidx] = dst_data_t{0};
                    }
                }
                for (int ic = 0; ic < IC; ic++) {
                    int iidx = n * IC + ic;
                    int widx = oc * IC + ic;
                    if (inDtype == AK_UINT8) {
                        dst_data[oidx] += param.scale * src_data[iidx] * weights_data[widx];
                    } else if (inDtype == AK_FLOAT) {
                        dst_data[oidx] += src_data[iidx] * weights_data[widx];
                    }

                }
            }
        }
        weights_data += OC * IC;
    }
}

using inprod_test_params_float = inprod_test_params;

template<DataType OpDataType,
         DataType InDataType,
         DataType BiasDataType,
         DataType OutDataType,
         typename LayOutType_op,
         typename LayOutType_in,
         typename LayOutType_bias,
         typename LayOutType_out>
bool _inner_product_test(inprod_test_params& p) {
    typedef Tensor<X86, OpDataType, LayOutType_op> opTensor;
    typedef Tensor<X86, InDataType, LayOutType_in> inTensor;
    typedef Tensor<X86, BiasDataType, LayOutType_bias> biasTensor;
    typedef Tensor<X86, OutDataType, LayOutType_out> outTensor;
    typedef typename opTensor::Dtype opDtype;
    typedef typename inTensor::Dtype inDtype;
    typedef typename outTensor::Dtype outDtype;

    // get icesword result
    Context<X86> ctx_host;

    test_inner_product_descr_t ipd = p.test_ipd;
    bool with_bias = p.with_bias;

    int total_ic = 0;
    for (int i = 0; i < ipd.ic.size(); i++) {
        total_ic += ipd.ic[i];
    }

    std::vector<inTensor *> inputs;
    for (int i = 0; i < ipd.ic.size(); i++) {
        Shape inputShape(ipd.mb, ipd.ic[i], ipd.kh, ipd.kw);
        inputs.push_back(new inTensor(inputShape));
        if (InDataType == AK_FLOAT) {
            fill_tensor_host_rand<Tensor4f>((Tensor4f&)*inputs[i]);
        } else if(InDataType == AK_UINT8) {
            fill_tensor_host_u8_rand<IoTensor4u8>((IoTensor4u8&)*inputs[i]);
        }
        // print_data<inTensor>(*inputs[i], "input :");
    }

    Shape weightShape(ipd.oc, total_ic, ipd.kh, ipd.kw);
    opTensor iceswordWeight(weightShape);
    if (InDataType == AK_FLOAT) {
        fill_tensor_host_rand<Tensor4f>((Tensor4f&)iceswordWeight);
    } else if (InDataType == AK_UINT8) {
        fill_tensor_host_s8_rand<OpTensor4i8>((OpTensor4i8&)iceswordWeight);
    }
    // print_data<opTensor>(iceswordWeight, "weight :");

    Shape biasShape(1, 1, 1, ipd.oc);
    biasTensor iceswordbias(biasShape);

    FcParam<opTensor> param(&iceswordWeight, &iceswordbias, ipd.oc, 1, false, p.scale);
    if (with_bias) {
        if (InDataType == AK_FLOAT) {
            fill_tensor_host_rand<Tensor4f>((Tensor4f&)iceswordbias);
        } else if(InDataType == AK_UINT8) {
            fill_tensor_host_rand<IoTensor4s32>((IoTensor4s32&)iceswordbias);
        }
    } else {
        param.bias_fp32 = nullptr;
        param.bias_s32 = nullptr;
    }

    Fc<X86, OpDataType, InDataType, OutDataType, LayOutType_op, LayOutType_in, LayOutType_out> iceswordFc;

    Shape outputShape(ipd.mb, 1, 1, ipd.oc);
    outTensor iceswordOutput, refOutput;
    vector<outTensor *> outputs(1, &iceswordOutput);
    if (OutDataType == AK_INT32) {
        param.scale = 1.0;
    }

    iceswordFc.compute_output_shape(inputs, outputs, param);
    iceswordOutput.re_alloc(iceswordOutput.shape());
    refOutput.re_alloc(iceswordOutput.shape());

    // get reference result
    compute_ref_inner_product_fwd<OpDataType, InDataType, OutDataType, LayOutType_op, LayOutType_in, LayOutType_out>(inputs, refOutput, param);

    iceswordFc.init(inputs, outputs, param, SPECIFY, VENDER_IMPL, ctx_host);
    iceswordFc(inputs, outputs, param, ctx_host);

    bool flag = false;
    flag = compare_tensor<outTensor>(iceswordOutput, refOutput, 1e-3);

    return flag;
}

template <DataType inDtype,
          DataType outDtype>
bool inner_product_test(inprod_test_params& p) {
    if (inDtype == AK_FLOAT) {
        return _inner_product_test<AK_FLOAT, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW, NCHW>(p);
    } else if (inDtype == AK_UINT8 && outDtype == AK_FLOAT) {
        return _inner_product_test<AK_INT8, AK_UINT8, AK_INT32, AK_FLOAT, NCHW, NHWC, NHWC, NHWC>(p);
    } else if (inDtype == AK_UINT8 && outDtype == AK_INT8) {
        return _inner_product_test<AK_INT8, AK_UINT8, AK_INT32, AK_INT8, NCHW, NHWC, NHWC, NHWC>(p);
    } else if (inDtype == AK_UINT8 && outDtype == AK_INT32) {
        return _inner_product_test<AK_INT8, AK_UINT8, AK_INT32, AK_INT32, NCHW, NHWC, NHWC, NHWC>(p);
    }
}

TEST(TestSaberFuncFcX86, test_vender_fc) {
    Env<X86>::env_init();

    inprod_test_params_float test_param[] = {
        { {10, {10, 10, 10}, 10, 1, 1}, false, 1.0 },
        { {10, {10, 10, 10}, 10, 1, 1}, true, 1.0 },

        { {50, {1024}, 1000, 1, 1}, false, 1.0 },            //MobileNet
        { {50, {2048}, 1000, 1, 1}, false, 1.0 },            //ResNet50
        { {50, {4096}, 1000, 1, 1}, false, 2.0 },
        { {50, {4096}, 4096, 1, 1}, false, 2.0 },
        { {50, {25088}, 4096, 1, 1}, false, 2.0 },           //VGG16

        { {50, {1024}, 1000, 1, 1}, true, 1.0 },             //MobileNet
        { {50, {2048}, 1000, 1, 1}, true, 1.0 },             //ResNet50
        { {50, {4096}, 1000, 1, 1}, true, 2.0 },
        { {50, {4096}, 4096, 1, 1}, true, 2.0 },
        { {50, {25088}, 4096, 1, 1}, true, 2.0 },            //VGG16
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        LOG(INFO) << "case " << i << ":";
        bool flag = false;

        flag = inner_product_test<AK_FLOAT, AK_FLOAT>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test f32f32f32 Passed";
        } else {
            LOG(ERROR) << "Test f32f32f32 Failed";
        }

        flag = inner_product_test<AK_UINT8, AK_INT32>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test u8s8s32 Passed";
        } else {
            LOG(ERROR) << "Test u8s8s32 Failed";
        }

        flag = inner_product_test<AK_UINT8, AK_FLOAT>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test u8s8f32 Passed";
        } else {
            LOG(ERROR) << "Test u8s8f32 Failed";
        }

        flag = inner_product_test<AK_UINT8, AK_INT8>(test_param[i]);
        if (flag) {
            LOG(INFO) << "Test u8s8s8 Passed";
        } else {
            LOG(ERROR) << "Test u8s8s8 Failed";
        }

    }
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}