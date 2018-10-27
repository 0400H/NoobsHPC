/* Copyright (c) 2016 NoobsDNN Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef NBDNN_ICESWORD_FUNCS_IMPL_X86_VENDER_LSTM_H
#define NBDNN_ICESWORD_FUNCS_IMPL_X86_VENDER_LSTM_H

#include "icesword/funcs/impl/x86/x86_common.h"
// #include "icesword/funcs/impl/impl_base.h"
#include "icesword/funcs/impl/impl_lstm.h"


namespace noobsdnn {
    namespace icesword {

        template<DataType OpDtype,
                 DataType inDtype,
                 DataType outDtype,
                 typename LayOutType_op,
                 typename LayOutType_in,
                 typename LayOutType_out>
            class VenderLstm<X86, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase<
            Tensor<X86, inDtype, LayOutType_in>,
            Tensor<X86, outDtype, LayOutType_out>,
            Tensor<X86, OpDtype, LayOutType_op>,
            LstmParam<Tensor<X86, OpDtype, LayOutType_op> >> {
            public:
                typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
                typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
                typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;

                typedef typename DataTensor_in::Dtype InDataType;
                typedef typename DataTensor_out::Dtype OutDataType;
                typedef typename OpTensor::Dtype OpDataType;

                VenderLstm() : avx2_available_(false),
                    max_thread_num_(1),
                    aligned_bias_(nullptr),
                    aligned_init_hidden_(nullptr) {}

                ~VenderLstm() {
                    if (weight_x_packed_.size()) {
                        for (int i = 0; i < weight_x_packed_.size(); i++) {
                            cblas_sgemm_free(weight_x_packed_[i]);
                        }
                    }

                    if (weight_h_packed_.size()) {
                        for (int i = 0; i < weight_h_packed_.size(); i++) {
                            cblas_sgemm_free(weight_h_packed_[i]);
                        }
                    }

                    if (this->aligned_bias_) {
                        zfree(this->aligned_bias_);
                        this->aligned_bias_ = nullptr;
                    }

                    if (this->aligned_init_hidden_) {
                        zfree(this->aligned_init_hidden_);
                        this->aligned_init_hidden_ = nullptr;
                    }

                }

                virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    LstmParam<OpTensor>& param,
                    Context<X86>& ctx) override;

                virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    LstmParam<OpTensor>& param,
                    Context<X86>& ctx) override;

                virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    LstmParam<OpTensor>& param) override;

            private:
                bool avx2_available_;
                int max_thread_num_;
                int word_size_;
                int hidden_size_;
                int aligned_hidden_size_;
                bool create_done = false;
                int direction_parallel_num_ = 2;
                int wave_front_thread_num_ = 1;
                int mkl_thread_num_ = 1;

                OpDataType *aligned_bias_;
                OpDataType *aligned_init_hidden_;
                OpDataType *aligned_init_hidden_c;

                std::vector<OpDataType *> weight_x_packed_;
                std::vector<OpDataType *> weight_h_packed_;
                std::vector<OpDataType *> aligned_wx_;
                std::vector<OpDataType *> aligned_wh_;

                DataTensor_out batched_h;
                DataTensor_out batched_c;
                DataTensor_in batched_x;
                DataTensor_in batched_x_reverse;
                DataTensor_out batched_xx;

                SaberStatus check_conf(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    LstmParam<OpTensor>& param);
                SaberStatus single_batch(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    LstmParam<OpTensor>& param);
        };

    } // namespace icesword
} // namespace noobsdnn
#endif // NBDNN_ICESWORD_FUNCS_IMPL_X86_VENDER_LSTM_H
