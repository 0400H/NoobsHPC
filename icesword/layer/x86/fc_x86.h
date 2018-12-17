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

#ifndef NBDNN_ICESWORD_LAYER_FC_X86_H
#define NBDNN_ICESWORD_LAYER_FC_X86_H

#include "icesword/layer/x86/common.h"

namespace noobsdnn {
namespace icesword {

template <DataType DType>
class Layer<X86, FC, DType> : public LayerBase<X86, Param<X86, FC>> {

public:
    typedef typename DataTrait<X86, DType>::Dtype Op_DType;

    Layer() {
        ws_ = nullptr;
    }

    ~Layer() {
        if (DType == DT_FLOAT) {
            for (auto mem : packed_weights) {
                    if (mem) {
                    cblas_sgemm_free((float *)mem);
                    mem = nullptr;
                }
            }
        } else {
            if (ws_) {
                zfree(ws_);
                ws_ = nullptr;
            }
        }
    }

    Status init(const std::vector<Tensor<X86> *> &inputs,
                std::vector<Tensor<X86> *> &outputs,
                Param<X86, FC> &param) override;

    Status create(const std::vector<Tensor<X86> *> &inputs,
                  std::vector<Tensor<X86> *> &outputs,
                  Param<X86, FC> &param) override;

    Status run(const std::vector<Tensor<X86> *> &inputs,
               std::vector<Tensor<X86> *> &outputs,
               Param<X86, FC> &param) override;

    Status forward(const std::vector<Tensor<X86> *> &inputs,
                   std::vector<Tensor<X86> *> &outputs,
                   Param<X86, FC> &param) override;

    Status backward(const std::vector<Tensor<X86> *> &inputs,
                    std::vector<Tensor<X86> *> &outputs,
                    Param<X86, FC> &param) override;

private:
    int batch_size;
    int output_channel;
    void *ws_;
    std::vector<float> scale;
    CBLAS_TRANSPOSE is_transpose_weights;
    std::vector<Op_DType *> packed_weights;

};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_LAYER_FC_X86_H