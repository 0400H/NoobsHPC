#include "fc_x86.h"

namespace noobsdnn {
namespace icesword {

template <>
Status Layer<X86, FC, DT_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
                                        std::vector<Tensor<X86> *>& outputs,
                                        Param<X86, FC> &param) {
    if (batch_size != 1) {
        int total_ic = 0;

        for (int i = 0; i < inputs.size(); i++) {
            int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());
            auto weight = static_cast<const float *>(param.weights->data()) + total_ic * output_channel;

            packed_weights.push_back(cblas_sgemm_alloc(CblasAMatrix, output_channel, batch_size, IC));
            cblas_sgemm_pack(CblasColMajor,
                             CblasAMatrix,
                             is_transpose_weights,
                             output_channel,
                             batch_size,
                             IC,
                             1.0,
                             weight,
                             IC,
                             packed_weights[i]);

            total_ic += IC;
        }
    }

    return S_Success;
}

template <>
Status Layer<X86, FC, DT_INT8>::create(const std::vector<Tensor<X86> *>& inputs,
                                       std::vector<Tensor<X86> *>& outputs,
                                       Param<X86, FC> &param) {
    if (ws_) {
        zfree(ws_);
        ws_ = nullptr;
    }
    ws_ = zmalloc(batch_size * output_channel * sizeof(int), 256);
    if (ws_ == nullptr) {
        return S_OutOfMem;
    }

    return S_Success;
}

template <>
Status Layer<X86, FC, DT_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
                                      std::vector<Tensor<X86> *>& outputs,
                                      Param<X86, FC> &param) {
    output_channel = outputs[0]->channel();
    batch_size = inputs[0]->count_valid(0, param.axis);

    is_transpose_weights = param.is_transpose_weights ?
                           CblasNoTrans :
                           CblasTrans;

    return create(inputs, outputs, param);
}

template <>
Status Layer<X86, FC, DT_INT8>::init(const std::vector<Tensor<X86> *>& inputs,
                                                std::vector<Tensor<X86> *>& outputs,
                                                Param<X86, FC> &param) {
    output_channel = outputs[0]->channel();
    batch_size = inputs[0]->count_valid(0, param.axis);

    for (int i = 0; i < output_channel; i ++) {
        scale.push_back((inputs[0]->get_scale()[0] * param.weights->get_scale()[i]) /
                        outputs[0]->get_scale()[0]);
    }

    is_transpose_weights = param.is_transpose_weights ?
                           CblasNoTrans :
                           CblasTrans;

    return create(inputs, outputs, param);
}

template <>
Status Layer<X86, FC, DT_FLOAT>::forward(const std::vector<Tensor<X86> *>& inputs,
                                         std::vector<Tensor<X86> *>& outputs,
                                         Param<X86, FC> &param) {
    auto dst = static_cast<float *>(outputs[0]->mutable_data());
    auto bias = param.bias ?
                static_cast<const float *>(param.bias->data()):
                nullptr;

    int total_ic = 0;
    for (int i = 0; i < inputs.size(); i++) {
        int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());

        auto src = static_cast<const float*>(inputs[i]->data());
        auto weight = static_cast<const float *>(param.weights->data()) + total_ic * output_channel;

        // C := alpha * op(A) * op(B) + beta * C
        if(i == 0) {
            if (batch_size == 1) {
                cblas_sgemm(CblasColMajor,                          // layout
                            is_transpose_weights,                   // a need to transpose or not
                            CblasNoTrans,                           // b need to transpose or not
                            output_channel,                         // m
                            batch_size,                             // n
                            IC,                                     // k
                            1.0,                                    // alpha
                            weight,                                 // a
                            IC,                                     // lda
                            src,                                    // b
                            IC,                                     // ldb
                            0.0,                                    // beta
                            dst,                                    // c
                            output_channel);                        // ldc
            } else {
                cblas_sgemm_compute(CblasColMajor,                  // layout
                                    CblasPacked,                    // a packed
                                    CblasNoTrans,                   // b need to transpose or not
                                    output_channel,                 // m
                                    batch_size,                     // n
                                    IC,                             // k
                                    packed_weights[i],              // a
                                    IC,                             // lda
                                    src,                            // b
                                    IC,                             // ldb
                                    0.0,                            // beta
                                    dst,                            // c
                                    output_channel);                // ldc
            }
        } else {
            if (batch_size == 1) {
                cblas_sgemm(CblasColMajor,
                            is_transpose_weights,
                            CblasNoTrans,
                            output_channel,
                            batch_size,
                            IC,
                            1.0,
                            weight,
                            IC,
                            src,
                            IC,
                            1.0,
                            dst,
                            output_channel);
            } else {
                cblas_sgemm_compute(CblasColMajor,
                                    CblasPacked,
                                    CblasNoTrans,
                                    output_channel,
                                    batch_size,
                                    IC,
                                    packed_weights[i],
                                    IC,
                                    src,
                                    IC,
                                    1.0,
                                    dst,
                                    output_channel);
            }
        }

        total_ic += IC;
    }
    if (bias) {
        #pragma omp parallel for schedule(static)
        for (int mb = 0; mb < batch_size; mb++) {
            cblas_saxpy(output_channel, 1.0, bias, 1.0, dst + mb * output_channel, 1);
        }
    }

    return S_Success;
}

template <>
Status Layer<X86, FC, DT_INT8>::forward(const std::vector<Tensor<X86> *>& inputs,
                                        std::vector<Tensor<X86> *>& outputs,
                                        Param<X86, FC> &param) {
    #define __FC_PARALLEL_FUNC [&](int mb, int oc) { \
        int dst_index = mb * output_channel + oc; \
        if (bias) { \
            dst[dst_index] = (scale[oc] == 1.f) ? \
                static_cast<int32_t *>(ws_)[dst_index] + bias[oc] : \
                scale[oc] * (static_cast<int32_t *>(ws_)[dst_index] + bias[oc]); \
        } else { \
            dst[dst_index] = (scale[oc] == 1.f) ? \
                dst[dst_index] = static_cast<int32_t *>(ws_)[dst_index] : \
                scale[oc] * static_cast<int32_t *>(ws_)[dst_index]; \
        } \
    }

    int c_offset = 0;
    int total_ic = 0;

    auto bias = param.bias ?
                static_cast<const int *>(param.bias->data()):
                nullptr;

    for (int i = 0; i < inputs.size(); i++) {
        int IC = inputs[i]->count_valid(param.axis, inputs[i]->dims());

        auto src = static_cast<const uint8_t *>(inputs[i]->data());
        auto weight = static_cast<const int8_t *>(param.weights->data()) + total_ic * output_channel;

        /* c = scale * { op(A) + a_offset_scale * a_offset } *
               { op(B) + b_offset_scale * b_offset } + beta * C + c_offset**/
        if (i == 0) {
            cblas_gemm_s8u8s32(CblasColMajor,                       // Layout
                               is_transpose_weights,                // a need to transpose or not
                               CblasNoTrans,                        // b need to transpose or not
                               CblasFixOffset,                      // c_offset_layout
                               output_channel,                      // m
                               batch_size,                          // n
                               IC,                                  // k
                               1.0,                                 // scale
                               weight,                              // a
                               IC,                                  // lda
                               0,                                   // a_offset
                               src,                                 // b
                               IC,                                  // ldb
                               0,                                   // b_offset
                               0.0,                                 // beta
                               static_cast<int *>(ws_),             // c
                               output_channel,                      // ldc
                               &c_offset);
        } else {
            cblas_gemm_s8u8s32(CblasColMajor,
                               is_transpose_weights,
                               CblasNoTrans,
                               CblasFixOffset,
                               output_channel,
                               batch_size,
                               IC,
                               1.0,
                               weight,
                               IC,
                               0,
                               src,
                               IC,
                               0,
                               1.0,
                               static_cast<int *>(ws_),
                               output_channel,
                               &c_offset);
        }

        total_ic += IC;
    }

    auto dst_dtype = outputs[0]->get_dtype();
    if (dst_dtype == DT_FLOAT) {
        auto dst = static_cast<float *>(outputs[0]->mutable_data());
        parallel_nd(batch_size, output_channel, __FC_PARALLEL_FUNC);
    } else if (dst_dtype == DT_INT32) {
        auto dst = static_cast<int32_t *>(outputs[0]->mutable_data());
        parallel_nd(batch_size, output_channel, __FC_PARALLEL_FUNC);
    } else if (dst_dtype == DT_INT8) {
        auto dst = static_cast<int8_t *>(outputs[0]->mutable_data());
        parallel_nd(batch_size, output_channel, __FC_PARALLEL_FUNC);
    } else {
        return S_UnImplError;
    }

    return S_Success;
}

template <>
Status Layer<X86, FC, DT_FLOAT>::backward(const std::vector<Tensor<X86> *> &inputs,
                                          std::vector<Tensor<X86> *> &outputs,
                                          Param<X86, FC> &param) {
    return S_UnImplError;
}

template <>
Status Layer<X86, FC, DT_INT8>::backward(const std::vector<Tensor<X86> *> &inputs,
                                         std::vector<Tensor<X86> *> &outputs,
                                         Param<X86, FC> &param) {
    return S_UnImplError;
}

template <>
Status Layer<X86, FC, DT_FLOAT>::run(const std::vector<Tensor<X86> *>& inputs,
                                     std::vector<Tensor<X86> *>& outputs,
                                     Param<X86, FC> &param) {
    switch (param.algorithm) {
        case FORWARD_FC_GEMM:
            return forward(inputs, outputs, param); break;
        case BACKWARD_FC_GEMM:
            return backward(inputs, outputs, param); break;
        default :
            return S_UnImplError;
    }
}

template <>
Status Layer<X86, FC, DT_INT8>::run(const std::vector<Tensor<X86> *>& inputs,
                                    std::vector<Tensor<X86> *>& outputs,
                                    Param<X86, FC> &param) {
    switch (param.algorithm) {
        case FORWARD_FC_GEMM:
            return forward(inputs, outputs, param); break;
        case BACKWARD_FC_GEMM:
            return backward(inputs, outputs, param); break;
        default :
            return S_UnImplError;
    }
}

} // namespace icesword
} // namespace noobsdnn