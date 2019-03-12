/* Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/Ldim_kENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "kernel/cpu_isa.h"
#include "convolution_x86.h"

namespace noobsdnn {
namespace icesword {

template <>
Status Operator<X86, CONVOLUTION, ET_forward_gemm, DT_FLOAT>::execute(
                                const std::vector<Tensor<X86> *>& inputs,
                                std::vector<Tensor<X86> *>& outputs,
                                ImplParam<X86, CONVOLUTION>& param) {
    auto src_ = static_cast<const float*>(inputs[0]->data());
    auto wei_ = static_cast<const float*>(weight_);
    auto dst_ = static_cast<float*>(outputs[0]->mutable_data());
    auto col_ = static_cast<float*>(column_);
    auto offset_ = static_cast<const float*>(bias_);

    #pragma omp parallel for collapse(2) num_threads(thread_num)
    for (int mb = 0; mb < batch; ++mb) {
        for (int g = 0; g < group; ++g) {
            auto thread_id = ice_get_thread_num();

            auto bias_stride = g * o_c;
            auto wei_stride = g * o_c * kh_kw_ic;
            auto src_stride = (mb * group + g) * ic_ih_iw;
            auto col_stride = thread_id * kh_kw_ic * oh_ow;
            auto dst_stride = (mb * group + g) * o_c * oh_ow;

            auto src_mem = src_ + src_stride;
            auto col_mem = col_ + col_stride;
            auto wei_mem = wei_ + wei_stride;
            auto dst_mem = dst_ + dst_stride;
            auto offset_mem = offset_ + bias_stride;

            if (with_img2col) {
                CHECK_EQ((column_ != nullptr), true) << "wrong column_ empty pointer !";
                img2col(src_mem, col_mem);
            }

            if (o_c == 1 || oh_ow == 1) {
                gemm.execute(wei_mem,                    // weight
                             with_img2col                // src
                             ? col_mem
                             : src_mem,
                             dst_mem,                    // dst
                             offset_mem,                 // offset
                             o_c,                        // M
                             oh_ow,                      // N
                             kh_kw_ic,                   // K
                             0,                          // offset_a
                             0,                          // offset_b
                             offset_mode,                // offset_mode
                             col_major,                  // col_major
                             trans_wei,                  // trans_a
                             trans_src,                  // trans_b
                             false,                      // pack_a
                             false,                      // pack_b
                             0.f,                        // beta
                             1.f);                       // if pack, alpha not work
            } else {
                gemm.execute(wei_pack_[g],               // weight
                             with_img2col                // src
                             ? col_mem
                             : src_mem,
                             dst_mem,                    // dst
                             offset_mem,                 // offset
                             o_c,                        // M
                             oh_ow,                      // N
                             kh_kw_ic,                   // K
                             0,                          // offset_a
                             0,                          // offset_b
                             offset_mode,                // offset_mode
                             col_major,                  // col_major
                             trans_wei,                  // trans_a
                             trans_src,                  // trans_b
                             true,                       // pack_a
                             false,                      // pack_b
                             0.f,                        // beta
                             1.f);                       // if pack, alpha not work
            }

            // LOG(ERROR) << " (wei_ + wei_stride)[0]:" << (wei_ + wei_stride)[0]
            //            << "\ncolumn_[0]:" << static_cast<const float*>(column_)[0]
            //            << "\n(dst_ + dst_stride)[0]:" << (dst_ + dst_stride)[0]
            //            << "\n(offset_ + bias_stride)[0]:" << (with_bias ? ((offset_ + bias_stride)[0]) : 0);
        }
    }

    // todo: some optimization
    if (algo_act == AT_relu) {
        relu_inference->execute(outputs, outputs, param.act_param);
    }

    return S_Success;
}

template <>
Status Operator<X86, CONVOLUTION, ET_forward_gemm, DT_INT8>::execute(
                                const std::vector<Tensor<X86> *>& inputs,
                                std::vector<Tensor<X86> *>& outputs,
                                ImplParam<X86, CONVOLUTION>& param) {

    return S_Success;
}

} // namespace icesword
} // namespace noobsdnn