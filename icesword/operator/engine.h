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

#ifndef NBDNN_ICESWORD_OPERATOR_ENGINE_H
#define NBDNN_ICESWORD_OPERATOR_ENGINE_H

#pragma once

#include <map>

// #include "icesword/types.h"

#include "icesword/operator/gemm.h"

#include "icesword/operator/x86/reference_x86.h"

#include "icesword/operator/x86/activation_x86.h"
#include "icesword/operator/x86/convolution_x86.h"
#include "icesword/operator/x86/pooling_x86.h"
#include "icesword/operator/x86/inner_product_x86.h"

#include "icesword/operator/x86/kernel/gemm_ref_x86.h"
#include "icesword/operator/x86/kernel/cblas_gemm_x86.h"

// namespace noobsdnn{
// namespace icesword{

/**
 * engine declear macro defination
**/

    // typedef Status (* funcptr) (const std::vector<Tensor<X86> *>& inputs,
    //                             std::vector<Tensor<X86> *>& outputs,
    //                             ImplParam<X86, IP>& param);
    // std::map<std::string, funcptr> engine = {
    //     {"inner_product_init_FLOAT_FLOAT", inner_product_init<DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT>.init},
    // };

    // engine["inner_product_init_FLOAT_FLOAT"](input, memory.outputs, impl_param);

// } // namespace icesword
// } // namespace noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_ENGINE_H