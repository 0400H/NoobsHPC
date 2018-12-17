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

#ifndef S_DNN_ICESWORD_LAYER_BASE_H
#define S_DNN_ICESWORD_LAYER_BASE_H

#include "icesword/tensor/tensor.h"

namespace noobsdnn {
namespace icesword {

template<TargetType TType, typename Param>
class LayerBase {

public:
    LayerBase() {}
    virtual ~LayerBase() {}

    virtual Status init(const std::vector<Tensor<TType> *> &inputs,
                        std::vector<Tensor<TType> *> &outputs, Param &param) = 0;

    virtual Status create(const std::vector<Tensor<TType> *> &inputs,
                          std::vector<Tensor<TType> *> &outputs, Param &param) = 0;

    virtual Status run(const std::vector<Tensor<TType> *> &inputs,
                       std::vector<Tensor<TType> *> &outputs, Param &param) = 0;

    virtual Status forward(const std::vector<Tensor<TType> *> &inputs,
                           std::vector<Tensor<TType> *> &outputs, Param &param) {
        return S_UnImplError;
    }

    virtual Status backward(const std::vector<Tensor<TType> *> &inputs,
                            std::vector<Tensor<TType> *> &outputs, Param &param) {
        return S_UnImplError;
    }

protected:

};


} // namespace icesword
} // namespace noobsdnn

#endif //S_DNN_ICESWORD_LAYER_BASE_H
