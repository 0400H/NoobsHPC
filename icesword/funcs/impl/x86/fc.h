/* Copyright (c) 2018 NoobsDNN, Anakin Authors All Rights Reserve.

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

#ifndef NBDNN_ICESWORD_FUNCS_IMPL_X86_ICESWORD_VENDER_FC_H
#define NBDNN_ICESWORD_FUNCS_IMPL_X86_ICESWORD_VENDER_FC_H

#include "icesword/funcs/impl/impl_fc.h"
#include "icesword/funcs/impl/x86/x86_common.h"

using namespace std;

namespace noobsdnn {
namespace icesword {

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class VenderFc<X86, OpDtype, AK_FLOAT, outDtype,
               LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase<
               Tensor<X86, AK_FLOAT, LayOutType_in>,
               Tensor<X86, outDtype, LayOutType_out>,
               Tensor<X86, OpDtype, LayOutType_op>,
               FcParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, AK_FLOAT, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> DataTensor_op;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_op::Dtype DataType_op;
    typedef typename DataTensor_out::Dtype DataType_out;

    VenderFc() {
        scale = 1.f;
        dst_tmp = nullptr;
    }

    ~VenderFc() {
        for (int i = packed_weights.size() - 1; i >= 0; i--) {
            if (packed_weights[i]) {
                cblas_sgemm_free(packed_weights[i]);
                packed_weights[i] = nullptr;
            }
        }

        if (dst_tmp) {
            zfree(dst_tmp);
            dst_tmp = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             FcParam<DataTensor_op> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               FcParam<DataTensor_op> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 FcParam<DataTensor_op> &param) override;

private:
    int MB;
    int OC;
    vector<DataType_op *> packed_weights;
    float scale;
    void  *dst_tmp;
};

template <DataType OpDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class VenderFc<X86, OpDtype, AK_UINT8, outDtype,
               LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase<
               Tensor<X86, AK_UINT8, LayOutType_in>,
               Tensor<X86, outDtype, LayOutType_out>,
               Tensor<X86, OpDtype, LayOutType_op>,
               FcParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, AK_UINT8, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> DataTensor_op;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_op::Dtype DataType_op;
    typedef typename DataTensor_out::Dtype DataType_out;

    VenderFc() {
        scale = 1.f;
        dst_tmp = nullptr;
    }

    ~VenderFc() {
        if (dst_tmp) {
            zfree(dst_tmp);
            dst_tmp = nullptr;
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             FcParam<DataTensor_op> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               FcParam<DataTensor_op> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 FcParam<DataTensor_op> &param) override;

private:
    int MB;
    int OC;
    float scale;
    void  *dst_tmp;
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_FUNCS_IMPL_X86_ICESWORD_VENDER_FC_H
