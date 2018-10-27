/* Copyright (c) 2018 NoobsDNN Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef NBDNN_ICESWORD_CORE_TYPES_H
#define NBDNN_ICESWORD_CORE_TYPES_H
#include <iostream>

namespace noobsdnn{

namespace icesword{

struct __true_type{};
struct __false_type{};
struct __invalid_type{};

//! target type
enum TargetTypeEnum {
    eINVALID = -1,
    eNV = 1,
    eAMD = 2,
    eARM = 3,
    eX86 = 4,
    eNVHX86 = 5,
    eNVHARM = 6,
    eARMGPU = 7,
    eARMDSP
};

template <TargetTypeEnum T>
struct TargetType {};
// NV device without pinned memory
typedef TargetType<eNV> NV;
typedef TargetType<eARM> ARM;
typedef TargetType<eARMGPU> ARMGPU;
typedef TargetType<eAMD> AMD;
typedef TargetType<eX86> X86;
// NV device with pinned memory
typedef TargetType<eNVHX86> NVHX86;
//typedef TargetType<eNVHARM> NVHARM;
// invalid target type, for target has only one memory block
typedef TargetType<eINVALID> INVLD;

//! target_type struct
struct W{};
struct HW{};
struct WH{};
struct NW{};
struct NHW{};
struct NCHW{};
struct NHWC{};
struct NCHW_C4{};
struct NCHW_C8{};
struct NCHW_C16{};
struct OIHW16I16O {};
struct GOIHW16I16O {};
struct WINO_FMT {};
//!target_category struct
struct _5D{};
struct _4D{};
struct _3D{};
struct _2D{};
struct _1D{};

enum DataType {
    AK_INVALID      =       -1,
    AK_HALF         =       0,
    AK_FLOAT        =       1,
    AK_DOUBLE       =       2,
    AK_INT8         =       3,
    AK_INT16        =       4,
    AK_INT32        =       5,
    AK_INT64        =       6,
    AK_UINT8        =       7,
    AK_UINT16       =       8,
    AK_UINT32       =       9,
    AK_STRING       =       10,
    AK_BOOL         =       11,
    AK_SHAPE        =       12,
    AK_TENSOR       =       13
};

typedef enum {
    SaberSuccess         = -1,                             /*!< No errors */
    SaberNotInitialized  = 1,                              /*!< Data not initialized. */
    SaberInvalidValue    = (1 << 1) + SaberNotInitialized, /*!< Incorrect variable value. */
    SaberMemAllocFailed  = (1 << 2) + SaberInvalidValue,   /*!< Memory allocation error. */
    SaberUnKownError     = (1 << 3) + SaberMemAllocFailed, /*!< Unknown error. */
    SaberOutOfAuthority  = (1 << 4) + SaberUnKownError,    /*!< Try to modified data not your own*/
    SaberOutOfMem        = (1 << 5) + SaberOutOfAuthority, /*!< OOM error*/
    SaberUnImplError     = (1 << 6) + SaberOutOfMem,       /*!< Unimplement error. */
    SaberWrongDevice     = (1 << 7) + SaberUnImplError     /*!< un-correct device. */
} SaberStatus;

typedef enum{
    STATIC = 1, /*!< choose impl by static policy */
    RUNTIME = 2, /*!< choose impl by compare performance at runtime */
    SPECIFY = 3,
    UNKNOWN = 4
}SaberImplStrategy;

typedef enum {
    nearest = 0,
    down
} round_mode;

typedef enum{
    Active_unknow = 0,
    Active_sigmoid = 1,
    Active_relu = 2,
    Active_tanh = 3,
    Active_clipped_relu = 4,
    Active_elu = 5,
    Active_identity = 6,
    Active_sigmoid_fluid = 7,
    Active_tanh_fluid = 8,
    Active_stanh = 9,
    Active_prelu = 10

} ActiveType;

typedef enum{
    Pooling_unknow = 0,
    Pooling_max = 1,
    Pooling_average_include_padding = 2,
    Pooling_average_exclude_padding = 3,
    Pooling_max_deterministic
} PoolingType;

typedef enum{
    Eltwise_unknow = 0,
    Eltwise_prod = 1,
    Eltwise_sum = 2,
    Eltwise_max = 3
} EltwiseType;

typedef enum{
    ACROSS_CHANNELS = 0,
    WITHIN_CHANNEL = 1
} NormRegion;

enum BoxCoderType {
    ENCODE_CENTER = 0,
    DECODE_CENTER = 1
};

enum CodeType {
    CORNER 		= 1,
    CENTER_SIZE = 2,
    CORNER_SIZE = 3
};

typedef enum {
    ICESWORD_POWER_HIGH = 0,
    ICESWORD_POWER_LOW  = 1,
    ICESWORD_POWER_FULL = 2
} PowerMode;

typedef enum {
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE
} BorderType;

typedef enum {
    PRIOR_MIN = 0,
    PRIOR_MAX = 1,
    PRIOR_COM = 2
} PriorType;

typedef enum {
    /** Undefined memory format, used for empty memory descriptors. */
    wino_undef = 0,
    /** Tensors of weights for 2x3 winograd convolutions. */
    wino_wei_aaOIoi,
    wino_wei_aaOio,
    wino_wei_aaOBiOo,
    /** Tensor of weights for 4x3 convolution. */
    wino_wei_OBaaIBOIio
} WinoFormat;

struct WinoDesc{
    WinoFormat wino_format;
    int r;
    int alpha;
    int ic;
    int oc;
    int ic_block;
    int oc_block;
    int ic2_block;
    int oc2_block;
    size_t size;
};

} //namespace icesword

} //namespace noobsdnn

#endif //NBDNN_ICESWORD_CORE_TYPES_H
