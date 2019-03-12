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

#ifndef NBDNN_ICESWORD_UTILS_H
#define NBDNN_ICESWORD_UTILS_H

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "icesword/types.h"

namespace noobsdnn{
namespace icesword{

#define ICESWORD_CHECK(condition) \
    do { \
    Status error = condition; \
    CHECK_EQ(error, S_Success) << " " << icesword_get_error_string(error); \
} while (0)

#define SPLICE_ARGS(ARG1, ARG2) ARG1##ARG2

#define ARRAY_SIZE(array) (sizeof(array) / sizeof(*array))

static inline const std::string get_layout_string(LayoutType layout) {
    switch(layout) {
        case LT_C :
            return std::string("LT_C");
        case LT_NC :
            return std::string("LT_NC");
        case LT_NGC :
            return std::string("LT_NGC");
        case LT_NCHW :
            return std::string("LT_NCHW");
        case LT_NHWC :
            return std::string("LT_NHWC");
        case LT_GOHWI :
            return std::string("LT_GOHWI");
        case LT_GOIHW :
            return std::string("LT_GOIHW");
        default:
            return std::string("undefined");
    }
    return std::string("undefined");
}

static inline const std::string get_io_dtype_string(DataType inDtype, DataType outDtype) {
    if (inDtype == DT_FLOAT && outDtype == DT_FLOAT) {
        return std::string("f32f32");
    } else if (inDtype == DT_FLOAT && outDtype == DT_UINT8) {
        return std::string("f32u8");
    } else if (inDtype == DT_UINT8 && outDtype == DT_FLOAT) {
        return std::string("u8f32");
    } else if (inDtype == DT_UINT8 && outDtype == DT_INT32) {
        return std::string("u8s32");
    } else if (inDtype == DT_UINT8 && outDtype == DT_UINT32) {
        return std::string("u8u32");
    } else if (inDtype == DT_UINT8 && outDtype == DT_INT8) {
        return std::string("u8s8");
    } else if (inDtype == DT_UINT8 && outDtype == DT_UINT8) {
        return std::string("u8u8");
    }
    return std::string("undefined");
}

static inline const std::string get_algorithm_string(AlgorithmType algo) {
    switch(algo) {
        case AT_nearest :
            return std::string("nearest");
        case AT_relu :
            return std::string("relu");
        case AT_leakyrelu :
            return std::string("leakyrelu");
        case AT_sigmoid :
            return std::string("sigmoid");
        case AT_tanh :
            return std::string("tanh");
        default:
            return std::string("undefined");
    }
}

static inline const std::string icesword_get_error_string(Status error_code){
    switch (error_code) {
        case S_Success:
            return std::string("STATUS_SUCCESS");
        case S_UnKownError:
            return std::string("STATUS_UNKNOWN_ERROR");
        case S_UnImplError:
            return std::string("STATUS_UNIMPL_ERROR");
        case S_NotInitialized:
            return std::string("STATUS_NOT_INITIALIZED");
        case S_InvalidValue:
            return std::string("STATUS_INVALID_VALUE");
        case S_MemAllocFailed:
            return std::string("STATUS_MEMALLOC_FAILED");
        default:
            return std::string("STATUS_UNDEFINED_ERRORS");
    }
}

static inline void gfree(void *ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

static inline void* gmalloc(size_t size, size_t alignment = 64) {
    void* ptr = nullptr;
    int rc = ::posix_memalign(&ptr, alignment, size);
    return (rc == 0) ? ptr : nullptr;
}

static inline void* gcalloc(size_t len, size_t size, size_t alignment = 64) {
    void* ptr = nullptr;
    int rc = ::posix_memalign(&ptr, alignment, len * size);
    if (rc == 0) {
        memset(ptr, 0, len * size);
        return ptr;
    }
    return nullptr;
}

class VectorPrint {
public:
    template <typename Dtype>
    static void print_float(Dtype* target) {
        float* f = (float*)target;
        printf("size = %d\n", sizeof(Dtype));

        for (int i = 0; i < sizeof(Dtype) / sizeof(float); i++) {
            printf(" %f ,", f[i]);
        }

        printf("\n");
    }
};

template<typename T>
static inline const T& min(const T& a, const T& b) {
    return a < b ? a : b;
}

template<typename T>
static inline const T& max(const T& a, const T& b) {
    return a > b ? a : b;
}

template <typename T, typename P>
static inline bool everyone_is(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
static inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
static inline bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
static inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
static inline bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}

template<typename T>
static inline void array_copy(T* dst, const T* src, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

template<typename T>
static inline bool array_cmp(const T* a1, const T* a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) {
            return false;
        }

    return true;
}

template<typename T, typename U>
static inline void array_set(T* arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        arr[i] = static_cast<T>(val);
    }
}

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_UTILS_H