/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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
#ifndef NBDNN_ICESWORD_FUNCS_CONV_ACT_CONV1X1_ACT_H
#define NBDNN_ICESWORD_FUNCS_CONV_ACT_CONV1X1_ACT_H

#include "icesword/core/tensor.h"
#include "icesword/funcs/funcs_utils.h"
#include "icesword/funcs/base.h"
#include "icesword/funcs/impl/impl_base.h"

#ifdef NVIDIA_GPU
#include "icesword/funcs/impl/cuda/conv_act.h"
#include "icesword/funcs/impl/cuda/vender_conv_act.h"
#endif

#ifdef USE_X86_PLACE
#include "icesword/funcs/impl/x86/conv_act_conv1x1_act.h"
#endif


namespace noobsdnn {
namespace icesword {

template <typename TargetType,
    DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op = NCHW,
    typename LayOutType_in = NHWC,
    typename LayOutType_out = NHWC
>
class ConvActConv1x1Act : public BaseFunc<
    Tensor<TargetType, inDtype, LayOutType_in>,
    Tensor<TargetType, outDtype, LayOutType_out>,
    Tensor<TargetType, OpDtype, LayOutType_op>,
    ImplBase,
    ConvActConv1x1ActParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            ConvActConv1x1ActParam>::BaseFunc;

    typedef TargetType targetType_t;
    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ConvActConv1x1ActParam<OpTensor> Param_t;
    typedef ImplBase<InDataTensor, OutDataTensor, OpTensor, Param_t> Impl_t;
    typedef const std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    ConvActConv1x1Act() = default;

    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v &output, Param_t& param) override {
        return SaberSuccess;
    }

    void update_weights(ConvActConv1x1ActParam<OpTensor> &param) { };

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case ICESWORD_IMPL:
                this->_impl.push_back(new SaberConvActConv1x1Act<TargetType, OpDtype, inDtype, outDtype,
                                                                 LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

    virtual SaberStatus init(Input_v& input, Output_v& output, Param_t& param,
                      SaberImplStrategy strategy, ImplEnum implenum,
                      Context<targetType_t > &ctx) override {

        return BaseFunc<Tensor<TargetType, inDtype, LayOutType_in>,
                Tensor<TargetType, outDtype, LayOutType_out>,
                Tensor<TargetType, OpDtype, LayOutType_op>,
                ImplBase,
                ConvActConv1x1ActParam>::init(input, output, param, strategy, implenum, ctx);
    }

private:

    virtual void pick_best_static() override {
        if (true) { // some condition?
            this->_best_impl = this->_impl[0];
        }
    }

    // virtual void pick_best_runtime(Input_v input, Output_v output, Param_t& param) override {}

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_FUNCS_CONV_ACT_CONV1X1_ACT_H
