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

#ifndef NBDNN_ICESWORD_OPERATOR_OPERATOR_H
#define NBDNN_ICESWORD_OPERATOR_OPERATOR_H

#pragma once

#include <vector>

#include "icesword/types.h"
#include "icesword/impl_param.h"
#include "icesword/core/tensor/tensor_op.h"

namespace noobsdnn{
namespace icesword{

template<TargetType TType, typename IMPLParam>
class OperatorBase {
public:
    OperatorBase() {}

    virtual ~OperatorBase() {}

    /* public: should to be overrided by public function */
    virtual Status release() = 0;
    virtual Status init(const std::vector<Tensor<TType> *> &src,
                        std::vector<Tensor<TType> *> &dst,
                        IMPLParam &impl_param) = 0;
    virtual Status execute(const std::vector<Tensor<TType> *> &src,
                           std::vector<Tensor<TType> *> &dst,
                           IMPLParam &impl_param) = 0;

    /* private: should to be overrided by private function */
    virtual Status init_check(const std::vector<Tensor<TType> *> &src,
                              std::vector<Tensor<TType> *> &dst,
                              IMPLParam &impl_param) {};

    virtual Status init_conf(const std::vector<Tensor<TType> *> &src,
                             std::vector<Tensor<TType> *> &dst,
                             IMPLParam &impl_param) {};

    virtual Status init_source(const std::vector<Tensor<TType> *> &src,
                               std::vector<Tensor<TType> *> &dst,
                               IMPLParam &impl_param) {};

};

template <TargetType target_type,
          OperatorType operator_type,
          ExecuteMethod execute_type,
          DataType operator_dtype = DT_FLOAT>
class Operator : public OperatorBase<target_type, ImplParam<target_type, operator_type>> {};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_OPERATOR_H
