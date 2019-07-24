/*  Copyright (c) 2018 NoobsHPC Authors All Rights Reserve.

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

#ifndef NBHPC_ICESWORD_OPERATOR_X86_CONV_H
#define NBHPC_ICESWORD_OPERATOR_X86_CONV_H

#include "icesword/operator/x86/common.h"

namespace noobshpc {
namespace icesword {

template <ExecuteMethod EType, DataType DType>
class Operator<X86, CONV, EType, DType> : public
      ImplBase<X86, ImplParam<X86, CONV>> {

public:
    typedef typename DataTrait<X86, DType>::Dtype OP_DType;

    Operator()
        : thread_num(1)
        , accept_(nullptr)
        , column_(nullptr)
        , bias_(nullptr)
        , weight_(nullptr)
        , relu_inference(nullptr)
    {}

    ~Operator() {
        release();
    }

    Status init(const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, CONV>& param) override;

    Status execute(const std::vector<Tensor<X86> *>& inputs,
                   std::vector<Tensor<X86> *>& outputs,
                   ImplParam<X86, CONV>& param) override;

    Status release() override;

private:
    CBLAS_GEMM<X86, DType> gemm;
    Operator<X86, ACT, FWD_REF, DType>* relu_inference;

    bool col_major;
    bool trans_src;
    bool trans_wei;
    bool with_bias;
    bool with_img2col;

    size_t thread_num;
    LayoutType layout;
    char offset_mode;
    std::vector<float> scale;


    void* accept_;
    void* column_;
    void* bias_;
    void* weight_;
    std::vector<OP_DType *> wei_pack_;

    size_t i_c;
    size_t o_c;
    size_t oh_ow;
    size_t ic_ih_iw;
    size_t kh_kw_ic;

    size_t batch;
    size_t group;
    size_t i_h;
    size_t i_w;
    size_t g_ic;
    size_t o_h;
    size_t o_w;
    size_t g_oc;
    size_t k_h;
    size_t k_w;
    size_t s_h;
    size_t s_w;
    size_t d_h;
    size_t d_w;
    size_t p_h;
    size_t p_w;
    std::string rm;
    std::string algo_act;

    Status init_check(const std::vector<Tensor<X86> *>& inputs,
                      std::vector<Tensor<X86> *>& outputs,
                      ImplParam<X86, CONV>& param) override;

    Status init_conf(const std::vector<Tensor<X86> *>& inputs,
                     std::vector<Tensor<X86> *>& outputs,
                     ImplParam<X86, CONV>& param) override;

    Status init_source(const std::vector<Tensor<X86> *>& inputs,
                       std::vector<Tensor<X86> *>& outputs,
                       ImplParam<X86, CONV>& param) override;

    Status img2col(const void *img, void *col);

};

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, CONV, EType, DType>::release() {
    if (accept_) {
        gfree(accept_);
        accept_ = nullptr;
    }
    if (column_) {
        gfree(column_);
        column_ = nullptr;
    }
    if (relu_inference) {
        gfree(relu_inference);
        relu_inference = nullptr;
    }
    for (auto & mem : wei_pack_) {
        if (mem) {
            gemm.release(mem);
            mem = nullptr;
        }
    }
    bias_ = nullptr;
    weight_ = nullptr;

    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, CONV, EType, DType>::init(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, CONV>& param) {
    if (init_check(inputs, outputs, param) != S_Success) {
        return S_UnImplError;
    }
    if (init_conf(inputs, outputs, param) != S_Success) {
        return S_UnImplError;
    }
    if (init_source(inputs, outputs, param) != S_Success) {
        return S_UnImplError;
    }

    return S_Success;
};

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, CONV, EType, DType>::init_check(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, CONV>& param) {
    if (inputs.size() == 0 ||
        outputs.size() == 0 ||
        inputs[0] == nullptr ||
        outputs[0] == nullptr ||
        inputs[0]->data() == nullptr ||
        outputs[0]->data() == nullptr ||
        param.get_weight() == nullptr ||
        param.get_weight()->data() == nullptr) {
        LOG(ERROR) << "wrong empty pointer !";
        return S_InvalidValue;
    }

    if (DType == DT_INT8) {
        if (inputs[0]->get_scale().size() == 0 ||
            outputs[0]->get_scale().size() == 0  ||
            param.get_weight()->get_scale().size() == 0 ) {
            LOG(ERROR) << "wrong scale size !";
            return S_InvalidValue;
        }
    }

    auto weight_shape = param.get_weight()->shape();
    auto g_oc = weight_shape[0];
    auto g_ic = weight_shape[3] * group;
    auto channel_check = g_ic % param.group
                       + g_oc % param.group;
    if ((group > 1) && (channel_check > 0)) {
        LOG(ERROR) << "wrong input or output channel !";
        return S_InvalidValue;
    }

    if (inputs[0]->get_layout() != outputs[0]->get_layout()) {
        LOG(ERROR) << "wrong input or output layout !";
        return S_InvalidValue;
    } else {
        if (inputs[0]->get_layout() != LT_NCHW &&
            inputs[0]->get_layout() != LT_NHWC) {
            LOG(ERROR) << "dont't support this layout !";
            return S_InvalidValue;
        }
    }

    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, CONV, EType, DType>::init_conf(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, CONV>& param) {
    with_bias = param.get_bias() ? true : false;
    offset_mode = with_bias ? 'C' : 'N';
    bias_ = with_bias ? param.get_bias()->data() : nullptr;
    weight_ = param.get_weight()->data();
    layout = inputs[0]->get_layout();

    group = param.group;
    s_h = param.stride_h;
    s_w = param.stride_w;
    d_h = param.dilation_h;
    d_w = param.dilation_w;
    p_h = param.pad_h;
    p_w = param.pad_w;
    rm = param.rm;
    algo_act = param.act_param.algo_act;

    auto input_shape = inputs[0]->shape();
    auto output_shape = outputs[0]->shape();
    auto weight_shape = param.get_weight()->shape();

    batch = input_shape[0];
    g_oc = weight_shape[0];
    k_h = weight_shape[1];
    k_w = weight_shape[2];
    g_ic = weight_shape[3] * group;
    if (layout == LT_NCHW) {
        col_major = false;
        trans_src = false;
        trans_wei = false;
        i_h = input_shape[2];
        i_w = input_shape[3];
    } else if (layout == LT_NHWC) {
        col_major = true;
        trans_src = true;
        trans_wei = true;
        i_h = input_shape[1];
        i_w = input_shape[2];
    }

    i_c = g_ic / group;
    o_c = g_oc / group;
    o_h = (i_h + 2 * p_h - k_h / d_h) / s_h + 1;
    o_w = (i_w + 2 * p_w - k_w / d_w) / s_w + 1;
    oh_ow = o_h * o_w;
    ic_ih_iw = i_c * i_h * i_w;
    kh_kw_ic = k_h * k_w * i_c;

    with_img2col = true;
    // with_img2col = !(o_h == i_h && o_w == i_w &&
    //                  k_h * k_w == 1 && group == 1);

    auto mb_g = batch * group;
    auto omp_max_threads = omp_get_max_threads();
    auto omp_mb_g_threads = mb_g < omp_max_threads ?
                            mb_g :
                            omp_max_threads;

    if (batch != 1) {
        thread_num = omp_mb_g_threads;
    } else {
        thread_num = mb_g > omp_max_threads / 2 ?
                     omp_mb_g_threads : 1;
    }

    if (algo_act == "relu") {
        relu_inference = new Operator<X86, ACT, FWD_REF, DType>;
    }

    #ifdef ICESWORD_VERBOSE
        auto io_layout = get_layout_string(layout);
        auto act_type = get_algorithm_string(algo_act);
        LOG(INFO) << "Convolution x86 verbose{"
                  << " layout:"     << io_layout
                  << " act:"        << act_type
                  << " bias:"       << (with_bias ? "true" : "false")
                  << " batch:"      << batch
                  << " group:"      << group
                  << " ic:"         << i_c
                  << " oc:"         << o_c
                  << " ih:"         << i_h
                  << " iw:"         << i_w
                  << " oh:"         << o_h
                  << " ow:"         << o_w
                  << " kh:"         << k_h
                  << " kw:"         << k_w
                  << " ph:"         << p_h
                  << " pw:"         << p_w
                  << " sh:"         << s_h
                  << " sw:"         << s_w
                  << " dh:"         << d_h
                  << " dw:"         << d_w
                  << " }";
    #endif

    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, CONV, EType, DType>::init_source(
                    const std::vector<Tensor<X86> *>& inputs,
                    std::vector<Tensor<X86> *>& outputs,
                    ImplParam<X86, CONV>& param) {
    if (EType == FWD_REF) {
        return S_Success;
    }

    // LOG(INFO) << thread_num << ' ' << kh_kw_ic << ' ' << oh_ow;
    column_ = gcalloc(thread_num * kh_kw_ic * oh_ow, sizeof(OP_DType));
    CHECK_EQ((column_ != nullptr), true) << "calloc memory failed !";

    if (o_c != 1 && oh_ow != 1) {
        for (auto g = 0; g < group; g++) {
            auto wei_ = static_cast<OP_DType *>(weight_) + g * o_c * kh_kw_ic;
            wei_pack_.push_back(static_cast<OP_DType *>(gemm.pack(wei_,      // ptr
                                                                  false,     // col_major
                                                                  true,      // packed_a
                                                                  false,     // need_trans
                                                                  o_c,       // M
                                                                  oh_ow,     // N
                                                                  kh_kw_ic,  // K
                                                                  1.f)));    // alpha
        }
    }

    if (DType == DT_FLOAT) {
    } else {
        accept_ = gcalloc(o_c * oh_ow, sizeof(int32_t));
        CHECK_EQ((accept_ != nullptr), true) << "calloc memory failed !";

        for (auto i = 0; i < o_c; i ++) {
            scale.push_back((inputs[0]->get_scale()[0] * param.get_weight()->get_scale()[i]) /
                            outputs[0]->get_scale()[0]);
        }
    }

    return S_Success;
}

template <ExecuteMethod EType, DataType DType>
Status Operator<X86, CONV, EType, DType>::img2col(const void *img,
                                                         void *col) {
    CHECK_EQ((img != nullptr), true) << "wrong empty pointer !";
    CHECK_EQ((col != nullptr), true) << "wrong empty pointer !";
    if (layout = LT_NCHW) {
        if (DType == DT_FLOAT) {
            auto src = static_cast<const float *>(img);
            auto dst = static_cast<float *>(col);

            #pragma omp parallel for collapse(4) num_threads(thread_num)
            for (auto ic = 0; ic < i_c; ++ic) {
                for (auto oh = 0; oh < o_h; ++oh) {
                    for (auto ow = 0; ow < o_w; ++ow) {
                        for (auto kh = 0; kh < k_h; ++kh) {
                            auto ih = oh * s_h - p_h + kh * d_h;
                            if (ih < 0 || ih >= i_h) continue;

                            #pragma omp simd
                            for (auto kw = 0; kw < k_w; ++kw) {
                                auto iw = ow * s_w - p_w + kw * d_w;
                                if (iw < 0 || iw >= i_w) continue;

                                int iidx = (ic * i_h + ih) * i_w + iw;
                                int didx = (((kh * k_w + kw) * i_c + ic) * o_h + oh) * o_w + ow;
                                dst[didx] = src[iidx];
                            }
                        }
                    }
                }
            }
        }

    } else { // LT_NHWC

    }

    return S_Success;
}

} // namespace icesword
} // namespace noobshpc

#endif // NBHPC_ICESWORD_OPERATOR_X86_CONV_H