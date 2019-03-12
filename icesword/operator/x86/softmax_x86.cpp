/* Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

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

#include "softmax_x86.h"

namespace noobsdnn {
namespace icesword {

template <>
Status Operator<X86, SOFTMAX, ET_forward_gemm, DT_FLOAT>::execute(
                        const std::vector<Tensor<X86> *>& inputs,
                        std::vector<Tensor<X86> *>& outputs,
                        ImplParam<X86, SOFTMAX>& param) {
    const OP_DType *src = nullptr;
    OP_DType *dst = nullptr;

    return S_Success;
}

} // namespace icesword
} // namespace noobsdnn
