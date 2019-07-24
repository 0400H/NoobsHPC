/* Copyright (c) 2018 NoobsHPC Authors, Inc. All Rights Reserved.

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

#ifndef TENSOR_OP_H
#define TENSOR_OP_H

#pragma once

#include <random>

#include "tensor.h"

namespace noobshpc{
namespace icesword{

/* example:
    fill_tensor_debug<DT_FLOAT>(inputs[0]->mutable_data(),
                                mb * g, ic/g * ih * iw, true, true);
*/
template <DataType DType>
Status fill_tensor_debug(void* matrix,
                         const size_t height,
                         const size_t width,
                         bool fill_hight_index = false,
                         bool with_print = false) {
    typedef typename DataTrait<X86, DType>::Dtype OP_DType;

    auto matrix_ = static_cast<OP_DType *>(matrix);
    if (matrix_ == nullptr) {
        LOG(ERROR) << "wrong matrix empty pointer !";
        return S_InvalidValue;
    }

    // without openmp to print data
    for (auto m = 0; m < height; ++m) {
        #pragma omp simd
        for (auto n = 0; n < width; ++n) {
            matrix_[m * width + n] = fill_hight_index ? m : n;
            if (with_print) {
                auto index = m * width + n;
                LOG(INFO) << "matrix" << "[" << index << "]: "
                          << (fill_hight_index ? m : n);
            }
        }
    }

    return S_Success;
}

template <typename Dtype>
void fill_tensor_const_func(Dtype* dio, Dtype value, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = value;
    }
}

template <typename Dtype>
void fill_tensor_rand_func(Dtype* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = static_cast<Dtype>(rand());
    }
}

template <typename Dtype>
void fill_tensor_rand_func(Dtype* dio, Dtype vstart, Dtype vend, long long size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    for (long long i = 0; i < size; ++i) {
        Dtype random_num = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
        dio[i] = random_num;
    }
}

template <typename Dtype>
void print_tensor(const Dtype* din, long long size, int width) {
    for (int i = 0; i < size; ++i) {
        printf("%.6f ", static_cast<float>(din[i]));
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <typename Dtype>
double tensor_mean_value(const Dtype* din, long long size) {
    double sum = 0.0;
    for (long long i = 0; i < size; ++i) {
        sum += din[i];
    }
    return sum / size;
}

/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param tensor  The reference of input tensor.
**/
template <TargetType TType>
static inline void fill_tensor_const(Tensor<TType>& tensor, float value) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case DT_UINT8: fill_tensor_const_func((unsigned char*)dio, static_cast<unsigned char>(value), size); break;
        case DT_INT8: fill_tensor_const_func((char*)dio, static_cast<char>(value), size); break;
        case DT_INT16: fill_tensor_const_func((short*)dio, static_cast<short>(value), size); break;
        case DT_UINT16: fill_tensor_const_func((unsigned short*)dio, static_cast<unsigned short>(value), size); break;
        case DT_HALF: fill_tensor_const_func((short*)dio, static_cast<short>(value), size); break;
        case DT_UINT32: fill_tensor_const_func((unsigned int*)dio, static_cast<unsigned int>(value), size); break;
        case DT_INT32: fill_tensor_const_func((int*)dio, static_cast<int>(value), size); break;
        case DT_FLOAT: fill_tensor_const_func((float*)dio, static_cast<float>(value), size); break;
        case DT_DOUBLE: fill_tensor_const_func((double*)dio, static_cast<double>(value), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

/**
 *  \brief Fill the tensor buffer with rand value.
 *  \param The reference of input tensor.
**/
template <TargetType TType>
void fill_tensor_rand(Tensor<TType>& tensor) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case DT_UINT8: fill_tensor_rand_func((unsigned char*)dio, size); break;
        case DT_INT8: fill_tensor_rand_func((char*)dio, size); break;
        case DT_INT16: fill_tensor_rand_func((short*)dio, size); break;
        case DT_UINT16: fill_tensor_rand_func((unsigned short*)dio, size); break;
        case DT_UINT32: fill_tensor_rand_func((unsigned int*)dio, size); break;
        case DT_INT32: fill_tensor_rand_func((int*)dio, size); break;
        case DT_HALF: fill_tensor_rand_func((short*)dio, size); break;
        case DT_FLOAT: fill_tensor_rand_func((float*)dio, size); break;
        case DT_DOUBLE: fill_tensor_rand_func((double*)dio, size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

/**
 *  \brief Fill the tensor buffer with rand value from vstart to vend.
 *  \param tensor The reference of input tensor.
**/
template <TargetType TType>
void fill_tensor_rand(Tensor<TType>& tensor, float vstart, float vend) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case DT_UINT8: fill_tensor_rand_func((unsigned char*)dio, static_cast<unsigned char>(vstart),
                                              static_cast<unsigned char>(vend), size); break;
        case DT_INT8: fill_tensor_rand_func((char*)dio, static_cast<char>(vstart), static_cast<char>(vend), size); break;
        case DT_INT16: fill_tensor_rand_func((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case DT_UINT16: fill_tensor_rand_func((unsigned short*)dio, static_cast<unsigned short>(vstart),
                                               static_cast<unsigned short>(vend), size); break;
        case DT_UINT32: fill_tensor_rand_func((unsigned int*)dio, static_cast<unsigned int>(vstart),
                                               static_cast<unsigned int>(vend), size); break;
        case DT_INT32: fill_tensor_rand_func((int*)dio, static_cast<int>(vstart), static_cast<int>(vend), size); break;
        case DT_HALF: fill_tensor_rand_func((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case DT_FLOAT: fill_tensor_rand_func((float*)dio, static_cast<float>(vstart), static_cast<float>(vend), size); break;
        case DT_DOUBLE: fill_tensor_rand_func((double*)dio, static_cast<double>(vstart), static_cast<double>(vend), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

/**
 *  \brief Print the data in host tensor.
 *  \param tensor  The reference of input tensor.
**/
template <TargetType TType>
void print_tensor(Tensor<TType>& tensor) {
    LOG(INFO) << "host tensor data:" << tensor.size();
    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();

    switch(type) {
        case DT_UINT8: print_tensor((const unsigned char*)data_ptr, size, width); break;
        case DT_INT8: print_tensor((const char*)data_ptr, size, width); break;
        case DT_UINT16: print_tensor((const unsigned short*)data_ptr, size, width); break;
        case DT_INT16: print_tensor((const short*)data_ptr, size, width); break;
        case DT_UINT32: print_tensor((const unsigned int*)data_ptr, size, width); break;
        case DT_INT32: print_tensor((const int*)data_ptr, size, width); break;
        case DT_FLOAT: print_tensor((const float*)data_ptr, size, width); break;
        case DT_DOUBLE: print_tensor((const double*)data_ptr, size, width); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    printf("\n");
}

/**
 *  \brief Print the valid data in host tensor.
 *  \param tensor  The reference of input tensor.
**/
template <TargetType TType>
void print_tensor_valid(Tensor<TType>& tensor) {
    LOG(INFO) << "host tensor data:" << tensor.valid_size();
    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();

    if (tensor.is_continue_mem()) {
        switch(type) {
            case DT_UINT8: print_tensor((const unsigned char*)data_ptr, size, width); break;
            case DT_INT8: print_tensor((const char*)data_ptr, size, width); break;
            case DT_UINT16: print_tensor((const unsigned short*)data_ptr, size, width); break;
            case DT_INT16: print_tensor((const short*)data_ptr, size, width); break;
            case DT_UINT32: print_tensor((const unsigned int*)data_ptr, size, width); break;
            case DT_INT32: print_tensor((const int*)data_ptr, size, width); break;
            case DT_FLOAT: print_tensor((const float*)data_ptr, size, width); break;
            case DT_DOUBLE: print_tensor((const double*)data_ptr, size, width); break;
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
        printf("\n");
    } else {
        Tensor<TType> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        print_tensor<TType>(tvalid);
    }
}
/**
 *  \brief compute mean value of the valid data in device tensor.
 *  \param tensor  The reference of input tensor.
**/
template <TargetType TType>
double tensor_mean_value(Tensor<TType>& tensor) {
    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    DataType type = tensor.get_dtype();

    switch (type) {
        case DT_UINT8: return tensor_mean_value((const unsigned char*)data_ptr, size);
        case DT_INT8: return tensor_mean_value((const char*)data_ptr, size);
        case DT_UINT16: return tensor_mean_value((const unsigned short*)data_ptr, size);
        case DT_INT16: return tensor_mean_value((const short*)data_ptr, size);
        case DT_UINT32: return tensor_mean_value((const unsigned int*)data_ptr, size);
        case DT_INT32: return tensor_mean_value((const int*)data_ptr, size);
        case DT_FLOAT: return tensor_mean_value((const float*)data_ptr, size);
        case DT_DOUBLE: return tensor_mean_value((const double*)data_ptr, size);
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    return 0.0;
}

/**
 *  \brief compute mean value of the valid data in device tensor.
 *  \param tensor  The reference of input tensor.
**/
template <TargetType TType>
double tensor_mean_value_valid(Tensor<TType>& tensor) {
    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    DataType type = tensor.get_dtype();

    if (tensor.is_continue_mem()) {
        switch (type) {
            case DT_UINT8: return tensor_mean_value((const unsigned char*)data_ptr, size);
            case DT_INT8: return tensor_mean_value((const char*)data_ptr, size);
            case DT_UINT16: return tensor_mean_value((const unsigned short*)data_ptr, size);
            case DT_INT16: return tensor_mean_value((const short*)data_ptr, size);
            case DT_UINT32: return tensor_mean_value((const unsigned int*)data_ptr, size);
            case DT_INT32: return tensor_mean_value((const int*)data_ptr, size);
            case DT_FLOAT: return tensor_mean_value((const float*)data_ptr, size);
            case DT_DOUBLE: return tensor_mean_value((const double*)data_ptr, size);
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
    } else {
        Tensor<TType> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        return tensor_mean_value<TType>(tvalid);
    }

    return 0.0;
}

template <typename Dtype>
void tensor_cmp(const Dtype* src1, const Dtype* src2, \
                     int size, double& max_ratio, double& max_diff) {
    const double eps = 1e-6f;
    max_diff = fabs(src1[0] - src2[0]);
    max_ratio = fabs(2.0 * max_diff / (src1[0] + src2[0] + eps));

    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1[i] - src2[i]);
        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = fabs(2.0 * max_diff / (src1[i] + src2[i] + eps));
        }
    }
}


} // namespace icesword
} // namespace noobshpc

#endif // TENSOR_OP_H
