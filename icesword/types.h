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
    LT_W            = 1,
    LT_HW           = 2,
    LT_WH           = 3,
    LT_NW           = 4,
    LT_NHW          = 5,
    LT_NCHW         = 6,
    LT_NHWC         = 7,
    LT_NCHW_C4      = 8,
    LT_NCHW_C8      = 9,
    LT_NCHW_C16     = 10,
    LT_OIHW16I16O   = 11,
    LT_GOIHW16I16O  = 12,
};

enum DataType {
    DT_INVALID      =       -1,
    DT_HALF         =       0,
    DT_FLOAT        =       1,
    DT_DOUBLE       =       2,
    DT_INT8         =       3,
    DT_INT16        =       4,
    DT_INT32        =       5,
    DT_INT64        =       6,
    DT_UINT8        =       7,
    DT_UINT16       =       8,
    DT_UINT32       =       9,
    DT_STRING       =       10,
    DT_BOOL         =       11,
    DT_SHAPE        =       12,
    DT_TENSOR       =       13,
};

enum Status{
    S_Success         = -1,                             /*!< No errors**/
    S_NotInitialized  = 1,                              /*!< Data not initialized.**/
    S_InvalidValue    = (1 << 1) + S_NotInitialized,    /*!< Incorrect variable value.**/
    S_MemAllocFailed  = (1 << 2) + S_InvalidValue,      /*!< Memory allocation error.**/
    S_UnKownError     = (1 << 3) + S_MemAllocFailed,    /*!< Unknown error.**/
    S_OutOfAuthority  = (1 << 4) + S_UnKownError,       /*!< Try to modified data not your own*/
    S_OutOfMem        = (1 << 5) + S_OutOfAuthority,    /*!< OOM error*/
    S_UnImplError     = (1 << 6) + S_OutOfMem,          /*!< Unimplement error.**/
};

struct Layout {
    virtual int num_index() {return -1;}
    virtual int channel_index() {return -1;}
    virtual int height_index() {return -1;}
    virtual int width_index() {return -1;}
    virtual int depth_index() {return -1;}
    virtual int inner_c() {return -1;}
    virtual int dims() {return -1;}
    virtual LayoutType type() {return LT_invalid;}
};
struct W : public Layout {
    int width_index() {return 0;}
    int dims() {return 1;}
    LayoutType type() {return LT_W;}
};
struct HW : public Layout {
    int height_index() {return 0;}
    int width_index() {return 1;}
    int dims() {return 2;}
    LayoutType type() {return LT_HW;}
};
struct WH : public Layout {
    int height_index() {return 1;}
    int width_index() {return 0;}
    int dims() {return 2;}
    LayoutType type() {return LT_WH;}
};
struct NW : public Layout {
    int num_index() {return 0;}
    int width_index() {return 1;}
    int dims() {return 2;}
    LayoutType type() {return LT_NW;}
};
struct NHW : public Layout {
    int num_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int dims() {return 3;}
    LayoutType type() {return LT_NHW;}
};
struct NCHW : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int dims() {return 4;}
    LayoutType type() {return LT_NCHW;}
};
struct NHWC : public Layout {
    int num_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int channel_index() {return 3;}
    int dims() {return 4;}
    LayoutType type() {return LT_NHWC;}
};
struct NCHW_C4 : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 4;}
    int dims() {return 5;}
    LayoutType type() {return LT_NCHW_C4;}
};
struct NCHW_C8 : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 8;}
    int dims() {return 5;}
    LayoutType type() {return LT_NCHW_C8;}
};
struct NCHW_C16 : public Layout {
    int num_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 16;}
    int dims() {return 5;}
    LayoutType type() {return LT_NCHW_C16;}
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_TYPES_H
