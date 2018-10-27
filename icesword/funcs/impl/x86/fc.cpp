#include "icesword/funcs/impl/x86/fc.h"


namespace noobsdnn {
namespace icesword {

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, AK_FLOAT, outDtype,
                     LayOutType_op, LayOutType_in, LayOutType_out> ::init(
                     const vector<DataTensor_in*>& inputs,
                     vector<DataTensor_out*>& outputs,
                     FcParam<DataTensor_op> &param, Context<X86> &ctx) {
    this->_ctx = &ctx;

    if (is_same<LayOutType_out, NCHW_C8>::value ||
        is_same<LayOutType_out, NCHW_C16>::value ||
        is_same<LayOutType_out, NHWC>::value) {
        if (inputs[0]->width() != 1 || inputs[0]->height() != 1) {
            LOG(ERROR) << "only support spatial size = 1 while NCHW_C16/NCHW_C8/NHWC layout";
            return SaberUnImplError;
        }
    }

    OC = outputs[0]->channel();
    MB = inputs[0]->count_valid(0, param.axis);

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, AK_UINT8, outDtype,
                     LayOutType_op, LayOutType_in, LayOutType_out> ::init(
                     const vector<DataTensor_in*>& inputs,
                     vector<DataTensor_out*>& outputs,
                     FcParam<DataTensor_op> &param, Context<X86> &ctx) {
    this->_ctx = &ctx;

    if (outDtype != AK_FLOAT && outDtype != AK_INT8 && outDtype != AK_INT32) {
        LOG(ERROR) << "Don't support output data type";
        return SaberUnImplError;
    }

    if (outDtype == AK_INT32 && scale != 1.0) {
        LOG(ERROR) << "Don't support output s32 withi non-1.0 scale";
        return SaberUnImplError;
    }

    OC = outputs[0]->channel();
    MB = inputs[0]->count_valid(0, param.axis);

    scale = param.scale;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, AK_FLOAT, outDtype,
                     LayOutType_op, LayOutType_in, LayOutType_out> ::create(
                     const vector<DataTensor_in*>& inputs,
                     vector<DataTensor_out*>& outputs,
                     FcParam<DataTensor_op> &param, Context<X86> &ctx) {
    this->_ctx = &ctx;
    this->_param = &param;

    if (MB != 1) {
        int total_IC {0};
        for (int i = 0; i < inputs.size(); i++) {
            int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());
            const DataType_out *weight = param.weights->data() + total_IC * OC;
            packed_weights.push_back(cblas_sgemm_alloc(CblasAMatrix, OC, MB, IC));

            // LOG(INFO) << "noobsdnn input[" << i << "] alloc passed";
            cblas_sgemm_pack(CblasColMajor,
                             CblasAMatrix,
                             param.is_transpose_weights ? CblasNoTrans : CblasTrans,
                             OC,
                             MB,
                             IC,
                             1.0,
                             weight,
                             IC,
                             packed_weights[i]);

            total_IC += IC;
        }
    }

    return SaberSuccess;
}

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, AK_UINT8, outDtype,
                     LayOutType_op, LayOutType_in, LayOutType_out> ::create(
                     const vector<DataTensor_in*>& inputs,
                     vector<DataTensor_out*>& outputs,
                     FcParam<DataTensor_op> &param, Context<X86> &ctx) {
    this->_ctx = &ctx;
    this->_param = &param;

    dst_tmp = zmalloc(MB * OC * sizeof(int), 256);

    return SaberSuccess;
}

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, AK_FLOAT, outDtype,
                     LayOutType_op, LayOutType_in, LayOutType_out> ::dispatch(
                     const vector<DataTensor_in*>& inputs,
                     vector<DataTensor_out*>& outputs,
                     FcParam<DataTensor_op> &param) {
    DataType_out *dst = outputs[0]->mutable_data();
    const float *bias_data = param.bias_fp32 ? param.bias_fp32->data() : nullptr;

    int total_IC {0};
    for (int i = 0; i < inputs.size(); i++) {
        const DataType_in *src = inputs[i]->data();
        const float *weight = param.weights->data() + total_IC * OC;
        int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());

        // C := alpha * op(A) * op(B) + beta * C
        if(i == 0) {
            if (MB == 1) {
                cblas_sgemm(CblasColMajor,                                                  // layout
                            param.is_transpose_weights ? CblasNoTrans : CblasTrans,         // a need to transpose or not
                            CblasNoTrans,                                                   // b need to transpose or not
                            OC,                                                             // m
                            MB,                                                             // n
                            IC,                                                             // k
                            1.0,                                                            // alpha
                            weight,                                                         // a
                            IC,                                                             // lda
                            src,                                                            // b
                            IC,                                                             // ldb
                            0.0,                                                            // beta
                            dst,                                                            // c
                            OC);                                                            // ldc
            } else {
                cblas_sgemm_compute(CblasColMajor,                                          // layout
                                    CblasPacked,                                            // a packed
                                    CblasNoTrans,                                           // b need to transpose or not
                                    OC,                                                     // m
                                    MB,                                                     // n
                                    IC,                                                     // k
                                    packed_weights[i],                                      // a
                                    IC,                                                     // lda
                                    src,                                                    // b
                                    IC,                                                     // ldb
                                    0.0,                                                    // beta
                                    dst,                                                    // c
                                    OC);                                                    // ldc
            }
        } else {
            if (MB == 1) {
                cblas_sgemm(CblasColMajor,
                            param.is_transpose_weights ? CblasNoTrans : CblasTrans,
                            CblasNoTrans,
                            OC,
                            MB,
                            IC,
                            1.0,
                            weight,
                            IC,
                            src,
                            IC,
                            1.0,
                            dst,
                            OC);
            } else {
                cblas_sgemm_compute(CblasColMajor,
                                    CblasPacked,
                                    CblasNoTrans,
                                    OC,
                                    MB,
                                    IC,
                                    packed_weights[i],
                                    IC,
                                    src,
                                    IC,
                                    1.0,
                                    dst,
                                    OC);
            }
        }

        total_IC += IC;
    }

    if (bias_data) {
        #pragma omp parallel for schedule(static)
        for (int mb = 0; mb < MB; mb++) {
            cblas_saxpy(OC,
                        1.0,
                        bias_data,
                        1.0,
                        dst + mb * OC,
                        1);
        }
    }

    // LOG(INFO) << "noobsdnn compute[" << i << "] passed";
    // LOG(INFO) << "inputs[]:dims: " << inputs[0]->dims();
    // LOG(INFO) << "inputs:size: " << inputs.size();
    // LOG(INFO) << "inputs:capacity: " << inputs.capacity();
    // LOG(INFO) << "output:size: " << outputs.size();
    // LOG(INFO) << "OC, MB, IC: " << OC << " "<< MB << " " << IC;

    return SaberSuccess;
}

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
SaberStatus VenderFc<X86, OpDtype, AK_UINT8, outDtype,
                     LayOutType_op, LayOutType_in, LayOutType_out> ::dispatch(
                     const vector<DataTensor_in*>& inputs,
                     vector<DataTensor_out*>& outputs,
                     FcParam<DataTensor_op> &param) {
    int c_offset {0};
    int total_IC {0};
    DataType_out *dst = outputs[0]->mutable_data();
    const int *bias_data = param.bias_s32 ? param.bias_s32->data() : nullptr;

    for (int i = 0; i < inputs.size(); i++) {
        const DataType_in *src = inputs[i]->data();
        const DataType_op *weight = param.weights->data() + total_IC * OC;
        int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());

        // C := scale * { op(A) + a_offset_scale * a_offset } * { op(B) + b_offset_scale * b_offset } + beta * C + c_offset
        if (i == 0) {
            cblas_gemm_s8u8s32(CblasColMajor,                                                   // Layout
                               param.is_transpose_weights ? CblasNoTrans : CblasTrans,          // a need to transpose or not
                               CblasNoTrans,                                                    // b need to transpose or not
                               CblasFixOffset,                                                  // c_offset_layout
                               OC,                                                              // m
                               MB,                                                              // n
                               IC,                                                              // k
                               1.0,                                                             // scale
                               weight,                                                          // a
                               IC,                                                              // lda
                               0,                                                               // a_offset
                               src,                                                             // b
                               IC,                                                              // ldb
                               0,                                                               // b_offset
                               0.0,                                                             // beta
                               static_cast<int *>(dst_tmp),                                     // c
                               OC,                                                              // ldc
                               &c_offset);
        } else {
            cblas_gemm_s8u8s32(CblasColMajor,
                               param.is_transpose_weights ? CblasNoTrans : CblasTrans,
                               CblasNoTrans,
                               CblasFixOffset,
                               OC,
                               MB,
                               IC,
                               1.0,
                               weight,
                               IC,
                               0,
                               src,
                               IC,
                               0,
                               1.0,
                               static_cast<int *>(dst_tmp),
                               OC,
                               &c_offset);
        }

        total_IC += IC;
    }

    utils::parallel_nd(MB, OC, [&](int i, int j) {
        int out_index = i * OC + j;
        int bias_index = j;
        if (bias_data) {
            if (scale == 1.0) {
                dst[out_index] = static_cast<int *>(dst_tmp)[out_index] + bias_data[bias_index];
            } else {
                dst[out_index] = scale * (static_cast<int *>(dst_tmp)[out_index] + bias_data[bias_index]);
            }
        } else {
            if (scale == 1.0) {
                dst[out_index] = static_cast<int *>(dst_tmp)[out_index];
            } else {
                dst[out_index] = scale * static_cast<int *>(dst_tmp)[out_index];
            }
        }
    });

    return SaberSuccess;
}

template class VenderFc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
template class VenderFc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW>;
template class VenderFc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C8, NCHW>;
template class VenderFc<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NHWC, NCHW>;
template class VenderFc<X86, AK_INT8, AK_UINT8, AK_FLOAT, NCHW, NHWC, NHWC>;
template class VenderFc<X86, AK_INT8, AK_UINT8, AK_INT8, NCHW, NHWC, NHWC>;
template class VenderFc<X86, AK_INT8, AK_UINT8, AK_INT32, NCHW, NHWC, NHWC>;

} // namespace icesword
} // namespace noobsdnn