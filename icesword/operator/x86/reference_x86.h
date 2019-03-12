/* Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LKENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef NBDNN_ICESWORD_OPERATOR_X86_REFERENCE_H
#define NBDNN_ICESWORD_OPERATOR_X86_REFERENCE_H

#include "common_x86.h"

namespace noobsdnn{
namespace icesword{

// template<typename targetType>
// void conv_int8(Tensor<targetType> &src, Tensor<targetType> &dst,
//                const char *weights, const auto *bias, auto group,
//                auto kernel_w, auto kernel_h, auto stride_w, auto stride_h,
//                auto dilation_w, auto dilation_h, auto pad_w, auto pad_h,
//                bool flag_bias, bool flag_relu, std::vector<float> &scale,
//                EltwiseImplParam<targetType> *elt_param = NULL,
//                float beta = 0.f, round_mode rm = nearest) {
//     auto src_data_uint8 = static_cast<const unsigned char*>(src.data());
//     auto src_data_int8 = static_cast<const char*>(src.data());
//     auto dst_data_uint8 = static_cast<unsigned char*>(dst.mutable_data());
//     auto dst_data_int8 = static_cast<char*>(dst.mutable_data());
//     auto weights_data = weights;
//     bool with_bias = flag_bias;
//     auto bias_data = bias;

//     auto batch = dst.num();
//     auto oc = dst.channel();
//     auto oh = dst.height();
//     auto ow = dst.width();

//     auto ic = src.channel();
//     auto ih = src.height();
//     auto iw = src.width();
//     auto oc = oc / group;
//     auto ic = ic / group;

//     float sum_scale = 1.f;
//     if (elt_param && (elt_param->operation == Eltwise_sum)) {
//         sum_scale = elt_param->coeff[1];
//     }

//     if (src.get_layout() == Layout_NHWC && dst.get_layout() == Layout_NHWC) {
//         #pragma omp parallel for num_threads(8) collapse(5) schedule(static)
//         for (auto n = 0; n < batch; ++n) {
//             for (auto oh = 0; oh < oh; ++oh) {
//                 for (auto ow = 0; ow < ow; ++ow) {
//                     for (auto g = 0; g < group; ++g) {
//                         for (auto oc = 0; oc < oc; ++oc) {
//                             auto out_idx = n * oh * ow * group * oc
//                                    + oh * ow * group * oc + ow * group * oc + g * oc + oc;
//                             float bias_d = with_bias ? (float)(bias_data[g * oc + oc]) : 0.f;
//                             float computing_v;
//                             if (dst.get_dtype() == DT_INT8) {
//                                 computing_v = bias_d + dst_data_int8[out_idx] * beta;
//                             } else {
//                                 computing_v = bias_d + dst_data_uint8[out_idx] * beta;
//                             }

//                             for (auto ic = 0; ic < ic; ++ic) {
//                                 for (auto kh = 0; kh < kernel_h; ++kh) {
//                                     for (auto kw = 0; kw < kernel_w; ++kw) {
//                                         auto iw = ow * stride_w - pad_w + kw * (dilation_w);
//                                         auto ih = oh * stride_h - pad_h + kh * (dilation_h);
//                                         if (iw < 0 || iw >= iw) continue;
//                                         if (ih < 0 || ih >= ih) continue;

//                                         auto iidx = n * ih * iw * ic
//                                                + ih * iw * group * ic
//                                                + iw * group * ic
//                                                + g * ic
//                                                + ic;
//                                         auto widx = g * oc * ic * kernel_h * kernel_w
//                                                + oc * ic * kernel_h * kernel_w
//                                                + ic * kernel_h * kernel_w
//                                                + kh * kernel_w
//                                                + kw;

//                                         if (src.get_dtype() == DT_INT8) {
//                                             computing_v += src_data_int8[iidx] * weights_data[widx];
//                                         }
//                                         else {
//                                             computing_v += src_data_uint8[iidx] * weights_data[widx];
//                                         }
//                                     }
//                                 }
//                             }
//                             computing_v = computing_v * scale[g * oc + oc];

//                             if (elt_param && (elt_param->operation == Eltwise_sum)) {
//                                 if (dst.get_dtype() == DT_INT8) {
//                                     computing_v += dst_data_int8[out_idx] * sum_scale;
//                                 } else {
//                                     computing_v += dst_data_uint8[out_idx] * sum_scale;
//                                 }
//                             }

//                             if (flag_relu) {
//                                 computing_v = computing_v > 0.f ? computing_v : 0.f;
//                             }

//                             if (dst.get_dtype() == DT_INT8) {
//                                 switch (rm) {
//                                     case nearest: dst_data_int8[out_idx] = saturate<int8_t>((int32_t)nearbyintf(computing_v)); break;
//                                     case down: dst_data_int8[out_idx] = saturate<int8_t>((int32_t)floorf(computing_v)); break ;
//                                 }
//                             } else {
//                                 switch (rm) {
//                                     case nearest: dst_data_uint8[out_idx] = saturate<uint8_t>((int32_t)nearbyintf(computing_v)); break;
//                                     case down: dst_data_uint8[out_idx] = saturate<uint8_t>((int32_t)floorf(computing_v)); break ;
//                                 }
//                             }
//                             // LOG(INFO) << "computing_v:" << computing_v << " scale[g*oc + oc]" << scale[g*oc + oc] << " out_idx:" << out_idx;
//                             // LOG(INFO) << "out_idx:" << out_idx << " dst_data[out_idx]:" << (int)dst_data[out_idx];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

template <typename src_dtype,
          typename wei_dtype,
          typename bias_dtype,
          typename dst_dtype,
          TargetType TType>
void conv_reference(const std::vector<Tensor<TType>* > &src,
                    std::vector<Tensor<TType>* > &dst,
                    ImplParam<TType, CONVOLUTION> &param) {
    auto with_bias = param.get_bias() == nullptr ? false : true;
    auto src_data = static_cast<const float *>(src[0]->data());
    auto weights_data = static_cast<const float *>(param.get_weight()->data());
    auto dst_data = static_cast<float *>(dst[0]->mutable_data());
    auto bias_data = with_bias
                   ? static_cast<const float *>(param.get_bias()->data())
                   : nullptr;

    auto batch = param.batch;
    auto group = param.group;
    auto i_h = param.in_height;
    auto i_w = param.in_width;
    auto i_gc = param.in_channel;
    auto i_c = i_gc / group;
    auto o_h = param.out_height;
    auto o_w = param.out_width;
    auto o_gc = param.out_channel;
    auto o_c = o_gc / group;
    auto k_h = param.kernel_h;
    auto k_w = param.kernel_w;
    auto p_h = param.pad_h;
    auto p_w = param.pad_w;
    auto s_h = param.stride_h;
    auto s_w = param.stride_w;
    auto d_h = param.dilation_h;
    auto d_w = param.dilation_w;
    auto algo_active = param.act_param.algo_active;
    auto layout = src[0]->get_layout();

    #ifdef ICESWORD_VERBOSE
        auto algo_act_string = get_algorithm_string(algo_active);
        auto layout_string = get_layout_string(layout);
        LOG(INFO) << "Convolution Verbose {"
                  << " bias:"         << (with_bias ? "true" : "false")
                  << " layout:"       << layout_string
                  << " algo_act:"     << algo_act_string
                  << " batch:"        << batch
                  << " group:"        << group
                  << " ic:"           << i_gc
                  << " ih:"           << i_h
                  << " iw:"           << i_w
                  << " oc:"           << o_gc
                  << " oh:"           << o_h
                  << " ow:"           << o_w
                  << " kh:"           << k_h
                  << " kw:"           << k_w
                  << " ph:"           << p_h
                  << " pw:"           << p_w
                  << " sh:"           << s_h
                  << " sw:"           << s_w
                  << " dh:"           << d_h
                  << " dw:"           << d_w
                  << " }";
    #endif

    if (src[0]->get_layout() == LT_NHWC && dst[0]->get_layout() == LT_NHWC) {
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
                            dst_data[out_idx] = with_bias ? bias_data[g * o_c + oc] : 0.f;

                            for (auto kh = 0; kh < k_h; ++kh) {
                                auto ih = oh * s_h - p_h + kh * d_h;
                                if (ih < 0 || ih >= i_h) continue;

                                for (auto kw = 0; kw < k_w; ++kw) {
                                    auto iw = ow * s_w - p_w + kw * d_w;
                                    if (iw < 0 || iw >= i_w) continue;

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
                                        dst_data[out_idx] += src_data[iidx] * weights_data[widx];
                                    }
                                }
                            }
                            auto accept = dst_data[out_idx];
                            switch(algo_active) {
                                case AT_relu :
                                    if (accept < 0.f) {
                                        dst_data[out_idx] = 0.f;
                                    }
                                    break;
                                case AT_sigmoid : // todo
                                    break;
                                default :
                                    break;
                            }
                        }
                    }
                }
            }
        }
    } else if (src[0]->get_layout() == LT_NCHW && dst[0]->get_layout() == LT_NCHW) {
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
                            dst_data[out_idx] = with_bias ? bias_data[g * o_c + oc] : 0.f;

                            for (auto kh = 0; kh < k_h; ++kh) {
                                auto ih = oh * s_h - p_h + kh * d_h;
                                if (ih < 0 || ih >= i_h) continue;

                                for (auto kw = 0; kw < k_w; ++kw) {
                                    auto iw = ow * s_w - p_w + kw * d_w;
                                    if (iw < 0 || iw >= i_w) continue;

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
                                        dst_data[out_idx] += src_data[iidx] * weights_data[widx];
                                    }
                                }
                            }
                            auto accept = dst_data[out_idx];
                            switch(algo_active) {
                                case AT_relu :
                                    if (accept < 0.f) {
                                        dst_data[out_idx] = 0.f;
                                    }
                                    break;
                                case AT_sigmoid : // todo
                                    break;
                                default :
                                    break;
                            }
                        }
                    }
                }
            }
        }
    } else {
        LOG(ERROR) << "Layout Error";
    }

}

template <typename src_dtype,
          typename dst_dtype,
          TargetType TType>
void pooling_reference(const std::vector<Tensor<TType>* > &src,
                       std::vector<Tensor<TType>* > &dst,
                       ImplParam<TType, POOLING> &param) {

    auto src_data = static_cast<const float *>(src[0]->data());
    auto dst_data = static_cast<float *>(dst[0]->mutable_data());

    auto layout = src[0]->get_layout();
    // auto batch = param.batch;
    // auto channel = param.channel;
    // auto i_h = param.in_height;
    // auto i_w = param.in_width;
    // auto o_h = param.out_height;
    // auto o_w = param.out_width;
    auto k_h = param.kernel_h;
    auto k_w = param.kernel_w;
    auto p_h = param.pad_h;
    auto p_w = param.pad_w;
    auto s_h = param.stride_h;
    auto s_w = param.stride_w;
    auto algo = param.algo;

    #ifdef ICESWORD_VERBOSE
        auto algo_string = get_algorithm_string(algo);
        auto layout_string = get_layout_string(layout);
        // LOG(INFO) << "Pooling Verbose {"
        //           << " layout"        << layout_string;
        //           << " algo:"         << algo_string
        //           << " batch:"        << batch
        //           << " channel:"      << channel
        //           << " ih:"           << i_h
        //           << " iw:"           << i_w
        //           << " oc:"           << o_gc
        //           << " oh:"           << o_h
        //           << " ow:"           << o_w
        //           << " kh:"           << k_h
        //           << " kw:"           << k_w
        //           << " sh:"           << s_h
        //           << " sw:"           << s_w
        //           << " ph:"           << p_h
        //           << " pw:"           << p_w
        //           << " }";
    #endif

    if (src[0]->get_layout() == LT_NHWC && dst[0]->get_layout() == LT_NHWC) {

    } else if (src[0]->get_layout() == LT_NCHW && dst[0]->get_layout() == LT_NCHW) {

    } else {
        LOG(ERROR) << "Layout Error";
    }

}

template <typename a_dtype,
          typename b_dtype,
          typename bias_dtype,
          typename c_dtype,
          TargetType TType>
void ip_reference(const std::vector<Tensor<TType>* > &src,
                  std::vector<Tensor<TType>* > &dst,
                  ImplParam<TType, INNERPRODUCT> &param) {
    auto N = dst[0]->channel();
    auto M = src[0]->batch();

    Shape OutShape({M, N}, LT_NC);
    Tensor<X86> c_tmp;
    c_tmp.re_alloc(OutShape, DT_INT32);

    bool with_active = param.with_active;
    auto with_bias = param.get_matrix_bias() != nullptr ? true : false;
    auto is_int8 = param.get_matrix_b()->get_dtype() == DT_INT8 ? true : false;
    auto accept_data = static_cast<int32_t *>(c_tmp.mutable_data());
    auto c_data = static_cast<c_dtype *>(dst[0]->mutable_data());
    auto b_data = static_cast<const b_dtype *>(param.get_matrix_b()->data());
    auto bias_data = with_bias ?
                     static_cast<const bias_dtype *>(param.get_matrix_bias()->data()) :
                     nullptr;

    for (auto i = 0; i < src.size(); i++) {
        auto K = src[i]->count_valid(1, src[i]->dims());
        auto a_data = static_cast<const a_dtype *>(src[i]->data());

        #ifdef ICESWORD_VERBOSE
            LOG(INFO) << "Inner Product Verbose {"
                      << " m: " << M
                      << " n: " << N
                      << " k: " << K
                      << " with_bias: " << (with_bias ? "true" : "false")
                      << " with_active: " << (with_active ? "true" : "false")
                      << " }";
        #endif

        #pragma omp parallel for collapse(2)
        for (auto m = 0; m < M; m++) {
            for (auto n = 0; n < N; n++) {
                auto oidx = m * N + n;
                if (i == 0) {
                    if (is_int8) {
                        accept_data[oidx] = c_dtype{0};
                    } else {
                        c_data[oidx] = c_dtype{0};
                    }
                }
                #pragma omp simd
                for (auto k = 0; k < K; k++) {
                    auto iidx = m * K + k;
                    auto widx = n * K + k;
                    if (is_int8) {
                        accept_data[oidx] += a_data[iidx] * b_data[widx];
                    } else {
                        c_data[oidx] += a_data[iidx] * b_data[widx];
                    }
                }
            }
        }
        b_data += N * K;
    }

    if (is_int8) {
        #pragma omp parallel for collapse(1)
        for (auto m = 0; m < M; m++) {
            #pragma omp simd
            for (auto n = 0; n < N; n++) {
                auto c_index = m * N + n;
                if (with_bias) {
                    accept_data[c_index] += bias_data[n];
                }
                float scale = (src[0]->get_scale()[0] * param.get_matrix_b()->get_scale()[n]) /
                              dst[0]->get_scale()[0];
                float acc_tmp = scale * accept_data[c_index];
                if (with_active && acc_tmp < 0) {
                    c_data[c_index] = 0;
                } else {
                    c_data[c_index] = (c_dtype)nearbyintf(acc_tmp);
                }
            }
        }
    } else {
        if (with_active || with_bias) {
            #pragma omp parallel for collapse(1)
            for (auto m = 0; m < M; m++) {
                #pragma omp simd
                for (auto n = 0; n < N; n++) {
                    auto c_index = m * N + n;
                    if (with_bias) {
                        c_data[c_index] += bias_data[n];
                    }
                    float c_tmp = c_data[c_index];
                    if (with_active && c_tmp < 0) {
                        c_data[c_index] = 0;
                    }
                }
            }
        }
    }
}

template <typename src_dtype,
          typename dst_dtype,
          TargetType TType>
void softmax_reference(const std::vector<Tensor<TType>* > &src,
                       std::vector<Tensor<TType>* > &dst,
                       ImplParam<TType, CONVOLUTION> &param) {
    auto with_bias = param.get_bias() == nullptr ? false : true;
    auto src_data = static_cast<const float *>(src[0]->data());
    auto dst_data = static_cast<float *>(dst[0]->mutable_data());

    auto layout = src[0]->get_layout();
    // auto batch = param.batch;
    // auto channel = param.channel;
    // auto i_h = param.in_height;
    // auto i_w = param.in_width;
    // auto o_h = param.out_height;
    // auto o_w = param.out_width;
    auto k_h = param.kernel_h;
    auto k_w = param.kernel_w;
    auto p_h = param.pad_h;
    auto p_w = param.pad_w;
    auto s_h = param.stride_h;
    auto s_w = param.stride_w;
    auto algo = param.algo;

    #ifdef ICESWORD_VERBOSE
        auto algo_string = get_algorithm_string(algo);
        auto layout_string = get_layout_string(layout);
        // LOG(INFO) << "Pooling Verbose {"
        //           << " layout"        << layout_string;
        //           << " algo:"         << algo_string
        //           << " batch:"        << batch
        //           << " channel:"      << channel
        //           << " ih:"           << i_h
        //           << " iw:"           << i_w
        //           << " oc:"           << o_gc
        //           << " oh:"           << o_h
        //           << " ow:"           << o_w
        //           << " kh:"           << k_h
        //           << " kw:"           << k_w
        //           << " sh:"           << s_h
        //           << " sw:"           << s_w
        //           << " ph:"           << p_h
        //           << " pw:"           << p_w
        //           << " }";
    #endif

    if (src[0]->get_layout() == LT_NHWC && dst[0]->get_layout() == LT_NHWC) {

    } else if (src[0]->get_layout() == LT_NCHW && dst[0]->get_layout() == LT_NCHW) {

    } else {
        LOG(ERROR) << "Layout Error";
    }

}

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_X86_REFERENCE_H