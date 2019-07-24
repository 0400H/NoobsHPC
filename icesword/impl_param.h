/* Copyright (c) 2018 NoobsHPC Authors, Inc. All Rights Reserved.
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

namespace noobshpc {
namespace icesword {

template <TargetType TType, OperatorType LType>
struct ImplParam {};

template <TargetType TType>
struct ImplParam <TType, ACT> {
    ImplParam() {};
    ImplParam(std::string act_algo, float leakyrelu_s = 0.01)
        : algo_act(act_algo)
        , leakyrelu_scale(leakyrelu_s)
    {}

    ImplParam(const ImplParam &right)
        : algo_act(right.algo_act)
        , leakyrelu_scale(right.leakyrelu_scale)
     {}

    ImplParam &operator=(const ImplParam &right) {
        this->algo_act = right.algo_act;
        this->leakyrelu_scale = right.leakyrelu_scale;
        return *this;
    }

    std::string algo_act;
    float leakyrelu_scale;
};

template <TargetType TType>
struct ImplParam <TType, CONV> {
    ImplParam() {};
    ImplParam(Tensor<TType>* weight_in,
              Tensor<TType>* bias_in,
              size_t group_in,
              size_t stride_h_in,
              size_t stride_w_in,
              size_t dilation_h,
              size_t dilation_w,
              size_t pad_h_in,
              size_t pad_w_in,
              std::string rm_in,
              ImplParam<TType, ACT> act_param_in)
        : weight(weight_in)
        , bias(bias_in)
        , group(group_in)
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
        , group(right.group)
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
        this->group = right.group;
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

    size_t group;
    size_t stride_h;
    size_t stride_w;
    size_t dilation_h;
    size_t dilation_w;
    size_t pad_h;
    size_t pad_w;
    std::string rm;
    ImplParam<TType, ACT> act_param;
private:
    Tensor<TType>* weight;
    Tensor<TType>* bias;
};

template <TargetType TType>
struct ImplParam <TType, AXPY> {
    ImplParam() {};
    ImplParam(Tensor<TType>* alpha_in,
              Tensor<TType>* bias_in)
        : alpha(alpha_in)
        , bias(bias_in)
    {}

    ImplParam(const ImplParam& right)
        : alpha(right.alpha)
        , bias(right.bias)
    {}

    ImplParam& operator=(const ImplParam& right) {
        this->alpha = right.alpha;
        this->bias = right.bias;
        return *this;
    }

    inline const Tensor<TType>* get_bias() {
        return bias;
    }

    inline const Tensor<TType>* get_alpha() {
        return alpha;
    }

private:
    Tensor<TType>* alpha;
    Tensor<TType>* bias;
};

} // icesword
} // noobshpc

#endif // ICESWORD_FUNCS_PARAM_H
