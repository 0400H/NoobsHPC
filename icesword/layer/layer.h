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

#ifndef NBDNN_ICESWORD_LAYER_LAYER_H
#define NBDNN_ICESWORD_LAYER_LAYER_H

#include <vector>

#include "noobsdnn_config.h"
#include "icesword/types.h"
#include "icesword/params.h"
#include "icesword/layer/base.h"
#include "icesword/tensor/tensor.h"


namespace noobsdnn{
namespace icesword{

/**
 * Layer declear macro defination
**/
#define DEFINE_LAYER_CLASS(layer_target, layer_type) \
    template <TargetType layer_target, \
              LayerType layer_type, \
              DataType layer_dtype = DT_FLOAT> \
    class Layer : public LayerBase<layer_target, Param<layer_target, layer_type> > {};


/**
 * Layer declearation
**/
DEFINE_LAYER_CLASS(X86, FC);

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_LAYER_LAYER_H
