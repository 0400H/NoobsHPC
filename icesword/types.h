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

namespace noobsdnn{
namespace icesword{

enum TargetType {
    TT_INVALID = -1,
    X86 = 1,
};

enum LayerType {
    Layer_invalid  = -1,
    FC = 1,
};

enum AlgorithmType {
    AT_invalid  = -1,
    FORWARD_FC_GEMM,
    BACKWARD_FC_GEMM,
};

enum LayoutType {
    LT_invalid      = -1,
    LT_C            = 1,
    LT_W            = 2,
    LT_NC           = 3,
    LT_NGC          = 4,
    LT_HW           = 5,
    LT_WH           = 6,
    LT_NHW          = 7,
    LT_NWH          = 8,
    LT_NCHW         = 9,
    LT_NHWC         = 10,
    LT_NCHW_C4      = 11,
    LT_NCHW_C8      = 12,
    LT_NCHW_C16     = 13,
    LT_GCHW         = 14,
    LT_GHWC         = 15,
    LT_NGCHW        = 16,
    LT_NGHWC        = 17,
    LT_NHWGC        = 18,
    LT_GNCHW        = 19,
    LT_GNHWC        = 20,
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
    S_NotInitialized  = 1,     // Data not initialized
    S_InvalidValue    = 2,     // Incorrect variable value
    S_MemAllocFailed  = 3,     // Memory allocation error
    S_UnKownError     = 4,     // Unknown error
    S_OutOfMem        = 5,     // OOM error*/
    S_UnImplError     = 6,     // Unimplement error
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_TYPES_H
