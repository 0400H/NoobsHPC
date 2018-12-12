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

#ifndef NBDNN_ICESWORD_FUNCS_IMPL_BASE_IMPL_H
#define NBDNN_ICESWORD_FUNCS_IMPL_BASE_IMPL_H

#include "icesword/core/context.h"
#include "icesword/core/tensor.h"

namespace noobsdnn {
namespace icesword {

    template<TargetType TType, typename Param>
    class LayerBase {
    public:

        LayerBase() {}
        virtual ~LayerBase() {}

        virtual SaberStatus init(const std::vector<Tensor<TType> *> &inputs,
                                 std::vector<Tensor<TType> *> &outputs,
                                 Param &param, Context<TType> &ctx) {
          return SaberUnImplError;
        }

        virtual SaberStatus create(const std::vector<Tensor<TType> *> &inputs,
                                   std::vector<Tensor<TType> *> &outputs,
                                   Param &param, Context<TType> &ctx) {
          return SaberUnImplError;
        }

        virtual SaberStatus run(const std::vector<Tensor<TType> *> &inputs,
                                     std::vector<Tensor<TType> *> &outputs,
                                     Param &param) {
          return SaberUnImplError;
        }

    protected:
        Param *_param;
        Context<TType> *_ctx;
    };

}
}
#endif //NBDNN_ICESWORD_FUNCS_IMPL_BASE_IMPL_H
