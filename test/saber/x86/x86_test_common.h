/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifndef ANAKIN_TEST_SABER_X86_TEST_COMMON_H
#define ANAKIN_TEST_SABER_X86_TEST_COMMON_H

#include <vector>
#include <assert.h>

#include "core/tensor.h"
#include "saber/core/tensor_op.h"
#include "saber/core/context.h"
#include "saber/saber_types.h"

#include "utils/logger/logger.h"

#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(*array))

using namespace std;
using namespace anakin::saber;

typedef TargetWrapper<X86> X86_API;
typedef Tensor<X86, AK_FLOAT, NCHW> Tensor4f;
typedef Tensor<X86, AK_UINT8, NCHW> Tensor4u8;
typedef Tensor<X86, AK_FLOAT, NCHW_C16> Tensor5f_C16;
typedef Tensor<X86, AK_FLOAT, NCHW_C8> Tensor5f_C8;
typedef Tensor<X86, AK_FLOAT, HW> Tensor2f;
typedef Tensor<X86, AK_FLOAT, W> Tensor1f;
typedef Tensor<X86, AK_INT8, NHWC> IoTensor4i8;
typedef Tensor<X86, AK_UINT8, NHWC> IoTensor4u8;
typedef Tensor<X86, AK_FLOAT, NHWC> IoTensor4f;
typedef Tensor<X86, AK_INT32, NHWC> IoTensor4s32;
typedef Tensor<X86, AK_INT32, NCHW> OpTensor4s32;
typedef Tensor<X86, AK_INT8, NCHW> OpTensor4i8;

template <typename T>
bool compare_tensor(T& data, T& ref_data, float eps = 1e-4) {
    typedef typename T::Dtype data_t;
    int flag = true;

    if (data.size() != ref_data.size()) {
        return false;
    }

    data_t absdiff = 0.f;
    data_t absref = 0.f;
    for (int i = 0; i < data.size(); i++) {
        absdiff = std::fabs(data.data()[i] - ref_data.data()[i]);
        absref = std::fabs(ref_data.data()[i]);
        float e = absdiff > eps ? absdiff / absref : absdiff;
        if (e <= eps) {
            // LOG(ERROR) << "out = " << data.data()[i] << ", out_ref = " << ref_data.data()[i];
            continue;
        } else {
            LOG(ERROR) << "index: " << i;
            LOG(ERROR) << "absdiff:" << absdiff;
            LOG(ERROR) << "out = " << data.data()[i] << ", out_ref = " << ref_data.data()[i];
            flag = false;
            break;
        }
    }
    return flag;
}

template <typename T>
void print_data(T& data, const char* str) {
    for (int i = 0; i < data.size(); i++) {
        int d = data.data()[i];
            LOG(INFO) << str << " index : " << i << " data = " << d;
        }
}

template <typename T>
bool compare_tensor_int(T& data, T& ref_data) {
    typedef typename T::Dtype data_t;
    int flag = true;

    if (data.size() != ref_data.size()) {
        return false;
    }
    for (int i = 0; i < data.size(); i++) {
        int d = data.data()[i];
        int ref_d = ref_data.data()[i];
        if (d != ref_d) {
            LOG(ERROR) << "index: " << i;
            LOG(ERROR) << "out = " << d << ", out_ref = " << ref_d;
            flag = false;
            break;
        }
        else {
            // LOG(ERROR) << "index: " << i;
            // LOG(ERROR) << "out = " << d << ", out_ref = " << ref_d;
        }
    }
    return flag;
}

enum LayoutType {
    Layout_Invalid = 0,
    Layout_NCHW = 1,
    Layout_NCHW_C16 = 2,
    Layout_NCHW_C8 = 3,
    Layout_NHWC = 4,
};

typedef struct _conv_act_params {
    LayoutType weight_type;
    LayoutType input_type;
    LayoutType output_type;
    int n, g;                                         // batch-size, group
    int ic, ih, iw;                                   // input-channel, input-height, input-width
    int oc, oh, ow;
    int kh, kw;                                       // kernel_height, kernel_width
    int pad_h, pad_w;                                 // padding
    int stride_h, stride_w;
    int dil_h, dil_w;                                 // dilation
    float alpha, beta;
    float negative_slope;
    float coef;
    bool with_bias;
    bool is_dw;
} conv_act_common_params;

#endif
