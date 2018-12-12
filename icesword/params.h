/* Copyright (c) 2018 NoobsDNN Authors, Inc. All Rights Reserved.
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

#ifndef NBDNN_ICESWORD_PARAMS_H
#define NBDNN_ICESWORD_PARAMS_H

#include <vector>

#include "icesword/types.h"
#include "icesword/tensor/tensor.h"

namespace noobsdnn {
namespace icesword {

template <TargetType TType, LayerType LType>
struct Param {};

template <TargetType TType>
struct Param <TType, FC> {
    Param() = delete;
    Param(Tensor<TType>* in_weight, Tensor<TType>* in_bias,
          int out_channel, AlgorithmType aalgorithm, bool wei_trans = false, int in_axis = 1) {
        oc = out_channel;
        bias = in_bias;
        axis = in_axis;
        weights = in_weight;
        algorithm = aalgorithm;
        is_transpose_weights = wei_trans;
    }
    Param(const Param& right) {
        oc = right.oc;
        bias = right.bias;
        axis = right.axis;
        weights = right.weights;
        algorithm = right.algorithm;
        is_transpose_weights = right.is_transpose_weights;
    }
    Param& operator=(const Param& right) {
        this->oc = right.oc;
        this->bias = right.bias;
        this->axis = right.axis;
        this->weights = right.weights;
        this->algorithm = right.algorithm;
        this->is_transpose_weights = right.is_transpose_weights;
        return *this;
    }
    bool operator==(const Param& right) {
        bool flag = (this->oc == right.oc) &&
                    (this->bias == right.bias) &&
                    (this->axis == right.axis) &&
                    (this->weights == right.weights) &&
                    (this->algorithm == right.algorithm) &&
                    (this->is_transpose_weights == right.is_transpose_weights);
        return flag;
    }

    int oc {0};
    int axis {1};
    bool is_transpose_weights {false};
    Tensor<TType>* weights {nullptr};
    Tensor<TType>* bias {nullptr};
    AlgorithmType algorithm {AT_invalid};
};

} // icesword
} // noobsdnn

#endif //ICESWORD_FUNCS_PARAM_H
