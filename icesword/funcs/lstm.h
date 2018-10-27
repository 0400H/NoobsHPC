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

#ifndef NBDNN_ICESWORD_FUNCS_LSTM_H
#define NBDNN_ICESWORD_FUNCS_LSTM_H

#include "icesword/funcs/base.h"
#include "icesword/funcs/impl/impl_base.h"
#ifdef NVIDIA_GPU
#include "icesword/funcs/impl/cuda/lstm.h"
//#include "icesword/funcs/impl/cuda/vender_lstm.h"
#endif

#ifdef USE_X86_PLACE
#include "icesword/funcs/impl/x86/lstm.h"
#endif

#ifdef USE_ARM_PLACE
#include "icesword/funcs/impl/impl_lstm.h"
#endif

namespace noobsdnn {
namespace icesword {

template<typename TargetType,
         DataType OpDtype,
         DataType inDtype = AK_FLOAT,
         DataType outDtype = AK_FLOAT,
         typename LayOutType_op = NCHW,
         typename LayOutType_in = NCHW,
         typename LayOutType_out = NCHW
         >
class Lstm : public BaseFunc <
    Tensor<TargetType, inDtype, LayOutType_in>,
    Tensor<TargetType, outDtype, LayOutType_out>,
    Tensor<TargetType, OpDtype, LayOutType_op>,
    ImplBase,
    LstmParam
    > {
public:
    using BaseFunc <
    Tensor<TargetType, inDtype, LayOutType_in>,
           Tensor<TargetType, outDtype, LayOutType_out>,
           Tensor<TargetType, OpDtype, LayOutType_op>,
           ImplBase,
           LstmParam >::BaseFunc;

    Lstm() = default;

    typedef Tensor<TargetType, inDtype, LayOutType_in> InDataTensor;
    typedef Tensor<TargetType, outDtype, LayOutType_out> OutDataTensor;
    typedef Tensor<TargetType, OpDtype, LayOutType_op> OpTensor;
    typedef LstmParam<OpTensor> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    // TODO:calc output shape
    virtual SaberStatus compute_output_shape(const Input_v& input, Output_v& output, \
            Param_t& param) override {
        int seqLength = input[0]->num();
        int hiddenSize=0;
        if(param.with_peephole){
            hiddenSize=param.bias()->valid_size()/7;
        } else{
            hiddenSize=param.bias()->valid_size()/4;
        }

        Shape output_shape = Shape(seqLength, hiddenSize, param.num_direction, 1);
        output[0]->set_seq_offset(input[0]->get_seq_offset());
        if(output.size()>=2){
            output[1]->set_seq_offset(input[0]->get_seq_offset());
        }
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case ICESWORD_IMPL:
                this->_impl.push_back(new SaberLstm<TargetType, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            case VENDER_IMPL:
                LOG(INFO) << "init vender";
                this->_impl.push_back(new VenderLstm<TargetType, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>);
                return SaberSuccess;
            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        //! lstm only has vendor implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_runtime(Input_v input, Output_v output, \
                                   Param_t& param, Context<TargetType>& ctx) override {
        //! lstm only has vendor implementation
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        //! lstm only has vendor implementation
        this->_best_impl = this->_impl[0];
    }

};

} // namespace icesword
} // namepace noobsdnn
#endif // NBDNN_ICESWORD_FUNCS_LSTM_H

