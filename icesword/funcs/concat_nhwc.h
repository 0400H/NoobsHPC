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
#ifndef NBDNN_ICESWORD_FUNCS_CONCAT_NHWC_H
#define NBDNN_ICESWORD_FUNCS_CONCAT_NHWC_H

#include "icesword/funcs/base.h"
#include "icesword/funcs/impl/impl_base.h"
#include "icesword/funcs/impl/x86/concat_nhwc.h"

namespace noobsdnn {
namespace icesword {

template<typename TargetType,
        DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out
>
class Concat_nhwc : public BaseFunc<
        Tensor<TargetType, inDtype, LayOutType_in>,
        Tensor<TargetType, outDtype, LayOutType_out>,
        Tensor<TargetType, OpDtype, LayOutType_op>,
        ImplBase,
        ConcatParam
> {
public:
    using BaseFunc<
            Tensor<TargetType, inDtype, LayOutType_in>,
            Tensor<TargetType, outDtype, LayOutType_out>,
            Tensor<TargetType, OpDtype, LayOutType_op>,
            ImplBase,
            ConcatParam>::BaseFunc;

    Concat_nhwc() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef ConcatParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {
        unsigned long input_size = input.size();

        Shape_v shapes_in;
        shapes_in.resize(input_size);
        //! get input size
        for (int i = 0; i < input_size; i++){
            shapes_in[i] = input[i]->valid_shape();
        }

        Shape shape_out = shapes_in[0];

        //! compute output shape
        for (int i = 1; i < input_size; ++i) {
            Shape sh = shapes_in[i];
            for (int j = 0; j < sh.dims(); ++j) {
                if (j == param.axis) { continue; }
                else if (sh[j] != -1) {
                            CHECK_EQ(shape_out[j], sh[j]) \
                        << "All inputs must have the same shape, except at concat_axis.";
                } else {
                    sh[j] = shape_out[j];
                    ICESWORD_CHECK(input[i]->set_shape(sh));
                }
            }
            shape_out[param.axis] += sh[param.axis];
        }
        return output[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case ICESWORD_IMPL_NHWC:
                this->_impl.push_back(new SaberConcat_nhwc <TargetType, OpDtype, inDtype, outDtype,
                LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        if (true) // some condition?
            this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};

} // namespace icesword
} // namespace noobsdnn


#endif // NBDNN_ICESWORD_FUNCS_CONCAT_H
