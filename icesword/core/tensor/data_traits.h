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

#ifndef DATA_TRAITS_H
#define DATA_TRAITS_H

#pragma once

#include "icesword/types.h"

namespace noobsdnn{
namespace icesword{

static size_t type_length(DataType type) {
    switch (type) {
    case DT_INT8:
        return 1;
    case DT_UINT8:
        return 1;
    case DT_INT16:
        return 2;
    case DT_UINT16:
        return 2;
    case DT_INT32:
        return 4;
    case DT_UINT32:
        return 4;
    case DT_INT64:
        return 8;
    case DT_HALF:
        return 2;
    case DT_FLOAT:
        return 4;
    case DT_DOUBLE:
        return 8;
    default:
        return 4;
    }
}

template <TargetType TType>
struct DataTraitBase {
    typedef void* PtrDtype;
};

template <TargetType TType, DataType datatype>
struct DataTrait {};

template <TargetType TType>
struct DataTrait<TType, DT_HALF> {
    typedef short Dtype;
    typedef short* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_FLOAT> {
    typedef float Dtype;
    typedef float* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_DOUBLE> {
    typedef double Dtype;
    typedef double* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_INT8> {
    typedef char Dtype;
    typedef char* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_INT16> {
    typedef short Dtype;
    typedef short* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_INT32> {
    typedef int Dtype;
    typedef int* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_INT64> {
    typedef long Dtype;
    typedef long* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_UINT8> {
    typedef unsigned char Dtype;
    typedef unsigned char* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_UINT16> {
    typedef unsigned short Dtype;
    typedef unsigned short* PtrDtype;
};

template <TargetType TType>
struct DataTrait<TType, DT_UINT32> {
    typedef unsigned int Dtype;
    typedef unsigned int* PtrDtype;
};


} // namespace icesword
} // namespace noobsdnn

#endif // DATA_TRAITS_H
