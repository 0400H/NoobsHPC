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

#ifndef NBDNN_ICESWORD_TENSOR_OP_H
#define NBDNN_ICESWORD_TENSOR_OP_H

#include "noobsdnn_config.h"
#include "icesword/core/tensor.h"
#include "icesword/core/context.h"

namespace noobsdnn{

namespace icesword{

const float eps = 1e-6f;

/**
* \brief reorder reorder tensors from src layout to dst layout
* \param src  source tensor reference
* \param dst  destination tensor reference
*/
template <class Tensor_s, class Tensor_d>
void reorder(Tensor_s& src, Tensor_d& dst);

/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param tensor  The reference of input tensor.
 */
template <TargetType TType>
void fill_tensor_const(Tensor<TType>& tensor, float value, typename Tensor<TType>::API::stream_t stream = NULL);


/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param The reference of input tensor.
 */
template <TargetType TType>
void fill_tensor_rand(Tensor<TType>& tensor, typename Tensor<TType>::API::stream_t stream = NULL);


/**
 *  \brief Fill the tensor buffer with rand value from vstart to vend.
 *  \param tensor The reference of input tensor.
 */
template <TargetType TType>
void fill_tensor_rand(Tensor<TType>& tensor, float vstart, float vend, typename Tensor<TType>::API::stream_t stream = NULL);

/**
 *  \brief Print the data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
template <TargetType TType>
void print_tensor(Tensor<TType>& tensor, typename Tensor<TType>::API::stream_t stream = NULL);

/**
 *  \brief Print the valid data in host tensor.
 *  \param tensor  The reference of input tensor.
 */
template <TargetType TType>
void print_tensor_valid(Tensor<TType>& tensor, typename Tensor<TType>::API::stream_t stream = NULL);

/**
 *  \brief compute mean value of the valid data in device tensor.
 *  \param tensor  The reference of input tensor.
 */
template <TargetType TType>
double tensor_mean_value(Tensor<TType>& tensor, typename Tensor<TType>::API::stream_t stream = NULL);

/**
 *  \brief compute mean value of the valid data in device tensor.
 *  \param tensor  The reference of input tensor.
 */
template <TargetType TType>
double tensor_mean_value_valid(Tensor<TType>& tensor, typename Tensor<TType>::API::stream_t stream = NULL);

template <typename Dtype >
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, int size, double& max_ratio, double& max_diff);

} // namespace icesword

} // namespace noobsdnn

#endif //NBDNN_ICESWORD_TENSOR_OP_H
