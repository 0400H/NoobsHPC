/* Copyright (c) 2018 NoobsHPC Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/Ldim_kENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "convolution.h"

namespace noobshpc {
namespace icesword {

template <>
Status Operator<X86, CONV, FWD_REF, DT_FLOAT>::execute(
                const std::vector<Tensor<X86> *>& inputs,
                std::vector<Tensor<X86> *>& outputs,
                ImplParam<X86, CONV>& param) {
    auto src_ = static_cast<const float*>(inputs[0]->data());
    auto wei_ = static_cast<const float*>(weight_);
    auto dst_ = static_cast<float*>(outputs[0]->mutable_data());
    auto offset_ = static_cast<const float*>(bias_);

    if (layout == LT_NHWC) {
        #pragma omp parallel for collapse(5)
        for (auto mb = 0; mb < batch; ++mb) {
            for (auto oh = 0; oh < o_h; ++oh) {
                for (auto ow = 0; ow < o_w; ++ow) {
                    for (auto g = 0; g < group; ++g) {
                        for (auto oc = 0; oc < o_c; ++oc) {
                            auto out_idx = mb * o_h * o_w * group * o_c
                                         + oh * o_w * group * o_c
                                         + ow * group * o_c
                                         + g * o_c
                                         + oc;
                            dst_[out_idx] = with_bias ? offset_[g * o_c + oc] : 0.f;

                            for (auto kh = 0; kh < k_h; ++kh) {
                                auto ih = oh * s_h + kh * d_h - p_h;
                                if (ih >= i_h) continue;

                                for (auto kw = 0; kw < k_w; ++kw) {
                                    auto iw = ow * s_w + kw * d_w - p_w;
                                    if (iw >= i_w) continue;

                                    #pragma omp simd
                                    for (auto ic = 0; ic < i_c; ++ic) {
                                        auto iidx = mb * i_h * i_w * group * i_c
                                                  + ih * i_w * group * i_c
                                                  + iw * group * i_c
                                                  + g * i_c
                                                  + ic;
                                        auto widx = g * o_c * k_h * k_w * i_c
                                                  + oc * k_h * k_w * i_c
                                                  + kh * k_w * i_c
                                                  + kw * i_c
                                                  + ic;
                                        dst_[out_idx] += src_[iidx] * wei_[widx];
                                    }
                                }
                            }
                            auto accept = dst_[out_idx];
                            if (algo_act == "relu") {
                                if (accept < 0.f) {
                                    dst_[out_idx] = 0.f;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (layout == LT_NCHW) {
        #pragma omp parallel for collapse(5)
        for (auto mb = 0; mb < batch; ++mb) {
            for (auto g = 0; g < group; ++g) {
                for (auto oc = 0; oc < o_c; ++oc) {
                    for (auto oh = 0; oh < o_h; ++oh) {
                        for (auto ow = 0; ow < o_w; ++ow) {
                            auto out_idx = mb * group * o_c * o_h * o_w
                                         + g * o_c * o_h * o_w
                                         + oc * o_h * o_w
                                         + oh * o_w
                                         + ow;
                            dst_[out_idx] = with_bias ? offset_[g * o_c + oc] : 0.f;

                            for (auto kh = 0; kh < k_h; ++kh) {
                                auto ih = oh * s_h + kh * d_h - p_h;
                                if (ih >= i_h) continue;

                                for (auto kw = 0; kw < k_w; ++kw) {
                                    auto iw = ow * s_w + kw * d_w - p_w;
                                    if (iw >= i_w) continue;

                                    #pragma omp simd
                                    for (auto ic = 0; ic < i_c; ++ic) {
                                        auto iidx = mb * group * i_c * i_h * i_w
                                                  + g * i_c * i_h * i_w
                                                  + ic * i_h * i_w
                                                  + ih * i_w
                                                  + iw;
                                        auto widx = g * o_c * k_h * k_w * i_c
                                                  + oc * k_h * k_w * i_c
                                                  + kh * k_w * i_c
                                                  + kw * i_c
                                                  + ic;
                                        dst_[out_idx] += src_[iidx] * wei_[widx];
                                    }
                                }
                            }
                            auto accept = dst_[out_idx];
                            if (algo_act == "relu") {
                                if (accept < 0.f) {
                                    dst_[out_idx] = 0.f;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        LOG(ERROR) << "Layout Error";
        return S_InvalidValue;
    }

    return S_Success;
}

template <>
Status Operator<X86, CONV, FWD_REF, DT_INT8>::execute(
                    const std::vector<Tensor<X86> *>& inputs,
                    std::vector<Tensor<X86> *>& outputs,
                    ImplParam<X86, CONV>& param) {

    return S_Success;
}

} // namespace icesword
} // namespace noobshpc