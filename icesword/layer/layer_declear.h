/* Copyright (c) 2018 NoobsDNN, Anakin Authors, Inc. All Rights Reserved.

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

#ifndef NBDNN_ICESWORD_FUNCS_IMPL_FC_H
#define NBDNN_ICESWORD_FUNCS_IMPL_FC_H

#include <vector>

#include "noobsdnn_config.h"
#include "types.h"
#include "params.h"
#include "layer_base.h"
#include "icesword/core/tensor.h"


namespace noobsdnn{
namespace icesword{

    #define DEFINE_LAYER_CLASS(layer_target, layer_type, run_type) \
        template <TargetType layer_target, \
                  LayerType layer_type, \
                  RunType run_type, \
                  DataType layer_dtype = AK_FLOAT> \
        class Layer : public LayerBase<layer_target, Param<layer_target, layer_type> > {};


    #define UNDEFINE_LAYER_TEMPLATE(layer_target, layer_type, run_type, layer_dtype) \
        template<> \
        SaberStatus Layer<layer_target, layer_type, run_type, layer_dtype>::create( \
                          const std::vector<Tensor<layer_target> *> &inputs, \
                          std::vector<Tensor<layer_target> *> &outputs, \
                          Param<layer_target, layer_type> &param, \
                          Context<layer_target> &ctx) {return SaberUnImplError;} \
        template<> \
        SaberStatus Layer<layer_target, layer_type, run_type, layer_dtype>::init( \
                          const std::vector<Tensor<layer_target> *> &inputs, \
                          std::vector<Tensor<layer_target> *> &outputs, \
                          Param<layer_target, layer_type> &param, \
                          Context<layer_target> &ctx) {return SaberUnImplError;} \
        template<> \
        SaberStatus Layer<layer_target, layer_type, run_type, layer_dtype>::run( \
                          const std::vector<Tensor<layer_target> *> &inputs, \
                          std::vector<Tensor<layer_target> *> &outputs, \
                          Param<layer_target, layer_type> &param) {return SaberUnImplError;}

     // DEFINE_LAYER_CLASS(layer_target, layer_type, layer_param);

    DEFINE_LAYER_CLASS(X86, FC, FORWARD);

}
}

#endif //NBDNN_ICESWORD_FUNCS_IMPL_FC_H
