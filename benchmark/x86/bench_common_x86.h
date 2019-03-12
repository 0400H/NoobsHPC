/* Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

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

#ifndef NBDNN_BENCHMARK_X86_COMMONH
#define NBDNN_BENCHMARK_X86_COMMONH

#pragma once

#include "icesword/types.h"
#include "icesword/utils.h"
#include "icesword/impl_param.h"
#include "icesword/core/timer.h"
#include "icesword/core/logger/logger.h"
#include "icesword/core/tensor/tensor_op.h"
#include "icesword/operator/engine.h"

#include <string>

using namespace noobsdnn::icesword;

template <OperatorType opType>
struct bench_operator_param {};

template <OperatorType opType>
struct benchmark_operator_param {};

template <TargetType TType,
          OperatorType OPType,
          ExecuteMethod EType,
          DataType OPDType = DT_FLOAT>
struct benchmark_operator_memory{
    Timer timer;
    Tensor<TType> weight;
    Tensor<TType> bias;
    ImplParam<TType, OPType> param;
    std::vector<float> weight_scale;
    std::vector<Tensor<TType>*> output;
    Operator<TType, OPType, EType, OPDType> op;
};

static inline void benchmark_timer(Timer timer, const std::string str = "") {
    LOG(INFO) << str << " backend average time: "
              << timer.get_time_ms(Timer::avg) << " ms";
    LOG(INFO) << str << " backend max time: "
              << timer.get_time_ms(Timer::max) << " ms";
    LOG(INFO) << str << " backend min time: "
              << timer.get_time_ms(Timer::min) << " ms";
}

template<typename op_mem_type>
static inline void benchmark_operator_timer(op_mem_type mem, const std::string str = "") {
    benchmark_timer(mem.timer, str);
}

// operator_mem_type should be reference type
template<typename operator_mem_type>
static inline void benchmark_operator_execute (
            std::vector<Tensor<X86>*>& input,
            operator_mem_type memory,
            const int loop = 1,
            const bool with_timer = false,
            const bool show_timer = false,
            const std::string str = "") {
    auto & op = memory.op;
    auto & param = memory.param;
    auto & timer = memory.timer;
    auto & output = memory.output;
    if (with_timer) {
        timer.clear();
        for (int i = 0; i < loop; i++) {
            timer.start();
            op.execute(input, output, param);
            timer.stop();
        }
        if (show_timer) {
            benchmark_operator_timer<operator_mem_type>(memory, str);
        }
    } else {
        op.execute(input, output, param);
    }
}

typedef struct {
    DataType input_dtype;
    DataType weight_dtype;
    DataType bias_dtype;
    DataType output_dtype;
} bench_datatype_param;

typedef struct {
    LayoutType input_ltype;
    LayoutType weight_ltype;
    LayoutType bias_ltype;
    LayoutType output_ltype;
} bench_layouttype_param;

typedef struct {
    size_t height;
    size_t width;
    size_t depth;
    size_t channel;
} bench_image_param;

template <> struct bench_operator_param<ACTIVATION> {
    AlgorithmType active_type;
};

template <> struct bench_operator_param<CONVOLUTION> {
    LayoutType layout;
    size_t batch, group, ih, iw, id, ic, oc,
           kh, kw, sh, sw, dh, dw, ph, pw;
    bool with_bias;
    bench_operator_param<ACTIVATION> act_param;
};

template <> struct bench_operator_param<INNERPRODUCT> {
    size_t M;
    size_t N;
    std::vector<size_t> K;
    bool with_bias;
    ExecuteMethod algo_type;
    bench_operator_param<ACTIVATION> act_param;
};

template <> struct benchmark_operator_param<ACTIVATION> {
    bench_operator_param<ACTIVATION> act_param;
    bench_datatype_param dtype_param;
};

template <> struct benchmark_operator_param<CONVOLUTION> {
    bench_operator_param<CONVOLUTION> conv_param;
    bench_datatype_param dtype_param;
};

template <> struct benchmark_operator_param<INNERPRODUCT> {
    bench_operator_param<INNERPRODUCT> ip_param;
    bench_datatype_param dtype_param;
};

#endif // NBDNN_BENCHMARK_X86_COMMONH