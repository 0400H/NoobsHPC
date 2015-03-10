/* Copyright (c) 2018 NoobsDNN, Anakin Authors, All Rights Reserved.

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

#ifndef NBDNN_ICESWORD_CORE_DATA_TRAITS_H
#define NBDNN_ICESWORD_CORE_DATA_TRAITS_H

#include "icesword/icesword_types.h"

namespace noobsdnn{

namespace icesword{

template <typename Ttype, DataType datatype>
struct DataTrait{
    typedef __invalid_type Dtype;
    typedef __invalid_type dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_HALF> {
    typedef short Dtype;
    typedef short dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_FLOAT> {
    typedef float Dtype;
    typedef float dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_DOUBLE> {
    typedef double Dtype;
    typedef double dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT8> {
    typedef char Dtype;
    typedef char dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT16> {
    typedef short Dtype;
    typedef short dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT32> {
    typedef int Dtype;
    typedef int dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_INT64> {
    typedef long Dtype;
    typedef long dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT8> {
    typedef unsigned char Dtype;
    typedef unsigned char dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT16> {
    typedef unsigned short Dtype;
    typedef unsigned short dtype;
};

template <typename Ttype>
struct DataTrait<Ttype, AK_UINT32> {
    typedef unsigned int Dtype;
    typedef unsigned int dtype;
};

} //namespace icesword

} //namespace noobsdnn

#endif //NBDNN_ICESWORD_CORE_DATA_TRAITS_H
