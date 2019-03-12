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

#ifndef NBDNN_ICESWORD_TYPES_H
#define NBDNN_ICESWORD_TYPES_H

#pragma once

namespace noobsdnn{
namespace icesword{

enum TargetType {
    TT_INVALID = -1,
    X86 = 1,
};

enum OperatorType {
    OP_invalid  = -1,
    AXPY,
    ACTIVATION,
    CONVOLUTION,
    ELTWISE,
    INNERPRODUCT,
    ARGMAX,
    ARGMEAN,
    ARGMIN,
    POOLING,
    SOFTMAX,
    LSTM,
    GRU,
};

enum ExecuteMethod {
    ET_invalid = -1,
    ET_default = 0,
    ET_forward_jit = 1,
    ET_forward_gemm = 2,
    ET_backward_jit = 3,
    ET_backward_gemm = 4,
};

enum AlgorithmType {
    AT_invalid = -1,
    AT_default = 0,

    // algorithm
    AT_max,
    AT_mean,
    AT_min,
    AT_nearest,
    AT_relu,
    AT_leakyrelu,
    AT_sigmoid,
    AT_tanh,
};

enum LayoutType {
    LT_invalid      = -1,
    LT_C            = 1,
    LT_NC           = 2,
    LT_HW           = 3,
    LT_NGC          = 4,
    LT_NCHW         = 5,
    LT_NHWC         = 6,
    LT_GOHWI        = 7,
    LT_GOIHW        = 8,
};

enum DataType {
    DT_INVALID      = -1,
    DT_HALF         = 0,
    DT_FLOAT        = 1,
    DT_DOUBLE       = 2,
    DT_INT8         = 3,
    DT_INT16        = 4,
    DT_INT32        = 5,
    DT_INT64        = 6,
    DT_UINT8        = 7,
    DT_UINT16       = 8,
    DT_UINT32       = 9,
    DT_STRING       = 10,
    DT_BOOL         = 11,
    DT_SHAPE        = 12,
    DT_TENSOR       = 13,
};

enum Status{
    S_Success         = -1,    // No errors
    S_UnKownError     = 1,     // Unknown error
    S_UnImplError     = 2,     // Unimplement error
    S_NotInitialized  = 3,     // Data not initialized
    S_InvalidValue    = 4,     // Incorrect variable value
    S_MemAllocFailed  = 5,     // Memory allocation error
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_TYPES_H
