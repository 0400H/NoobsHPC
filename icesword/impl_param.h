/* Copyright (c) 2018 NoobsDNN Authors, Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" bASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef NbDNN_ICESWORD_PARAMS_H
#define NbDNN_ICESWORD_PARAMS_H

#pragma once

#include <vector>

#include "icesword/types.h"
#include "icesword/core/tensor/tensor.h"

namespace noobsdnn {
namespace icesword {

template <TargetType TType, OperatorType LType>
struct ImplParam {};

template <TargetType TType>
struct ImplParam <TType, ACTIVATION> {
    ImplParam() {};
    ImplParam(AlgorithmType act, float leakyrelu_s = 0.01)
        : algo_active(act)
        , leakyrelu_scale(leakyrelu_s)
    {}

    ImplParam(const ImplParam &right) : algo_active(right.algo_active) {}

    ImplParam &operator=(const ImplParam &right) {
        this->algo_active = right.algo_active;
        this->leakyrelu_scale = right.leakyrelu_scale;
        return *this;
    }

    AlgorithmType algo_active;
    float leakyrelu_scale;
};

template <TargetType TType>
struct ImplParam <TType, CONVOLUTION> {
    ImplParam() {};
    ImplParam(Tensor<TType>* weight_in,
              Tensor<TType>* bias_in,
              size_t batch_in,
              size_t group_in,
              size_t height_in,
              size_t width_in,
              size_t channel_in,
              size_t height_out,
              size_t width_out,
              size_t channel_out,
              size_t kernel_h_in,
              size_t kernel_w_in,
              size_t stride_h_in,
              size_t stride_w_in,
              size_t dilation_h,
              size_t dilation_w,
              size_t pad_h_in,
              size_t pad_w_in,
              AlgorithmType rm_in,
              ImplParam<TType, ACTIVATION> act_param_in)
        : weight(weight_in)
        , bias(bias_in)
        , batch(batch_in)
        , group(group_in)
        , in_height(height_in)
        , in_width(width_in)
        , in_channel(channel_in)
        , out_height(height_out)
        , out_width(width_out)
        , out_channel(channel_out)
        , kernel_h(kernel_h_in)
        , kernel_w(kernel_w_in)
        , stride_h(stride_h_in)
        , stride_w(stride_w_in)
        , dilation_h(dilation_h)
        , dilation_w(dilation_w)
        , pad_h(pad_h_in)
        , pad_w(pad_w_in)
        , act_param(act_param_in)
        , rm(rm_in)
    {}

    ImplParam(const ImplParam& right)
        : weight(right.weight)
        , bias(right.bias)
        , batch(right.batch)
        , group(right.group)
        , in_height(right.in_height)
        , in_width(right.in_width)
        , in_channel(right.in_channel)
        , out_height(right.out_height)
        , out_width(right.out_width)
        , out_channel(right.out_channel)
        , kernel_h(right.kernel_h)
        , kernel_w(right.kernel_w)
        , stride_h(right.stride_h)
        , stride_w(right.stride_w)
        , dilation_h(right.dilation_h)
        , dilation_w(right.dilation_w)
        , pad_h(right.pad_h)
        , pad_w(right.pad_w)
        , act_param(right.act_param)
        , rm(right.rm)
    {}

    ImplParam& operator=(const ImplParam& right) {
        this->weight = right.weight;
        this->bias = right.bias;
        this->batch = right.batch;
        this->group = right.group;
        this->in_height = right.in_height;
        this->in_width = right.in_width;
        this->in_channel = right.in_channel;
        this->out_height = right.out_height;
        this->out_width = right.out_width;
        this->out_channel = right.out_channel;
        this->kernel_h = right.kernel_h;
        this->kernel_w = right.kernel_w;
        this->stride_h = right.stride_h;
        this->stride_w = right.stride_w;
        this->dilation_h = right.dilation_h;
        this->dilation_w = right.dilation_w;
        this->pad_h = right.pad_h;
        this->pad_w = right.pad_w;
        this->act_param = right.act_param;
        this->rm = right.rm;
        return *this;
    }

    inline const Tensor<TType>* get_weight() {
        return weight;
    }

    inline const Tensor<TType>* get_bias() {
        return bias;
    }

    size_t batch;
    size_t group;
    size_t in_height;
    size_t in_width;
    size_t in_channel;
    size_t out_height;
    size_t out_width;
    size_t out_channel;
    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h;
    size_t stride_w;
    size_t dilation_h;
    size_t dilation_w;
    size_t pad_h;
    size_t pad_w;
    AlgorithmType rm;
    ImplParam<TType, ACTIVATION> act_param;
private:
    Tensor<TType>* weight;
    Tensor<TType>* bias;
};

template <TargetType TType>
struct ImplParam <TType, POOLING> {
    ImplParam() {};
    ImplParam(size_t kernel_h_in,
              size_t kernel_w_in,
              size_t stride_h_in,
              size_t stride_w_in,
              size_t pad_h_in,
              size_t pad_w_in,
              AlgorithmType algo_pool)
        : kernel_h(kernel_h_in)
        , kernel_w(kernel_w_in)
        , stride_h(stride_h_in)
        , stride_w(stride_w_in)
        , pad_h(pad_h_in)
        , pad_w(pad_w_in)
        , algo(algo_pool)
    {}

    ImplParam(const ImplParam& right)
        : kernel_h(right.kernel_h)
        , kernel_w(right.kernel_w)
        , stride_h(right.stride_h)
        , stride_w(right.stride_w)
        , pad_h(right.pad_h)
        , pad_w(right.pad_w)
        , algo(right.algo)
    {}

    ImplParam& operator=(const ImplParam& right) {
        this->kernel_h = right.kernel_h;
        this->kernel_w = right.kernel_w;
        this->stride_h = right.stride_h;
        this->stride_w = right.stride_w;
        this->pad_h = right.pad_h;
        this->pad_w = right.pad_w;
        this->algo = right.algo;
        return *this;
    }

    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h;
    size_t stride_w;
    size_t pad_h;
    size_t pad_w;
    AlgorithmType algo;
private:
};

template <TargetType TType>
struct ImplParam <TType, INNERPRODUCT> {
    ImplParam() {};
    ImplParam(Tensor<TType>* matrix_b,
              Tensor<TType>* matrix_bias,
              bool a_trans = false,
              bool b_trans = false,
              bool with_active = false)
        : Matrix_B(matrix_b)
        , Matrix_Offset(matrix_bias)
        , trans_a(a_trans)
        , trans_b(b_trans)
        , with_active(with_active)
    {}

    ImplParam(const ImplParam& right)
        : Matrix_B(right.Matrix_B)
        , Matrix_Offset(right.Matrix_Offset)
        , trans_a(right.trans_a)
        , trans_b(right.trans_b)
        , with_active(right.with_active)
    {}

    ImplParam& operator=(const ImplParam& right) {
        this->Matrix_B = right.Matrix_B;
        this->Matrix_Offset = right.Matrix_Offset;
        this->trans_a = right.trans_a;
        this->trans_b = right.trans_b;
        this->with_active = right.with_active;
        return *this;
    }

    inline const Tensor<TType>* get_matrix_b() {
        return Matrix_B;
    }

    inline const Tensor<TType>* get_matrix_bias() {
        return Matrix_Offset;
    }

    bool trans_a;
    bool trans_b;
    bool with_active;
private:
    Tensor<TType>* Matrix_B;
    Tensor<TType>* Matrix_Offset;
};

} // icesword
} // noobsdnn

#endif // ICESWORD_FUNCS_PARAM_H
