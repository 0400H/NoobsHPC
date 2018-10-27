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

#ifndef NBDNN_ICESWORD_FUNCS_IMPL_X86_JIT_CALL_CONF_H
#define NBDNN_ICESWORD_FUNCS_IMPL_X86_JIT_CALL_CONF_H

#include "icesword/funcs/impl/x86/x86_common.h"

namespace noobsdnn {
namespace icesword {
namespace jit {

// convolution
enum conv_version_t {ver_unused, ver_fma, ver_avx512_core, ver_4fma, ver_4vnni, ver_vnni};
enum conv_loop_order_t { loop_cgn, loop_gnc, loop_ngc };
enum conv_kernel_kind_t {embd_bcast, expl_bcast};

enum conv_1x1_loop_order_t {
    loop_rbl, loop_rlb, loop_lbr, loop_lrb, loop_blr,
    loop_brl
};

enum {
    FLAG_MB_FIRST = 1 << 0, FLAG_MB_LAST = 1 << 1,
    FLAG_OC_FIRST = 1 << 2, FLAG_OC_LAST = 1 << 3,
    FLAG_IC_FIRST = 1 << 4, FLAG_IC_LAST = 1 << 5,
    FLAG_SP_FIRST = 1 << 6, FLAG_SP_LAST = 1 << 7,
    FLAG_REDUCE_FIRST = 1 << 8, FLAG_REDUCE_LAST = 1 << 9,
};

struct jit_1x1_conv_call_t {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *bias_data; // used in forward and backward_weights only
    const void *acc_s32;
    const void *scales;

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t output_stride; // used in backward_weights only

    size_t reduce_pos_flag;
};

struct jit_wino_conv_call_t {
    const void *src;
    const void *dst;
    const void *wei;
    const void *dst_b;
};

struct jit_wino_conv_dst_trans_call_t {
    const void *wino_dst;
    const void *dst;
    const void *v_y_masks;
    const void *v_x_masks;
    const void *bias;
    const void *scales;
};

struct jit_wino_conv_src_trans_call_t {
    const void *src;
    const void *wino_src;
    const void *v_y_masks;
    const void *v_x_masks;
};

struct jit_fconv_call_t {
    const void *src;
    const void *dst; /* hack, non-const for forward */
    const void *wei;
    const void *bias;
    const void *scales;
    const void *acc_s32;

    const void *wei1x1;
    const void *bias1x1;
    const void *acc1x1;
    const void *scales1x1;
    size_t ocb3x3;

    size_t kh_padding;
    size_t channel;
};

struct jit_conv_conf_2x3_wino_t {
    conv_version_t ver;

    int m;
    int r;
    int alpha;
    int tile_h, tile_w;

    int mb;
    int ngroups, ic, oc;
    int ih, iw, oh, ow;
    int l_pad, t_pad;
    int r_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;

    int nb_ic, ic_block;
    int nb_oc, oc_block;

    int w_block_size, h_block_size;

    DataType bia_dt;
    DataType dst_dt;

    int is_oc_scale;
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;

    bool with_bias, with_relu;
    float relu_negative_slope;
    bool with_sum;
    bool small_mb;

    int xb, yb;
    int inp_stride;
    int out_stride;
    int wei_stride;
    int bia_stride;

    int M, N, K;
    int m_block, n_block, k_block;
    int n2_block, n_chunks;
    int k2_block, k_chunks;

    round_mode rm;
    float sum_scale;
};

struct jit_fconv_conf_t {
    int mb;
    int gp, ic, oc;
    int ih, iw, oh, ow;
    int kh, kw;
    int sh, sw;
    int l_pad, t_pad;  // left, top padding
    int ic_block, oc_block;
    int nb_ic, nb_oc;
    // @note: nc_ic==(nb_ic_blocking * ic_chunk)
    int nb_ic_blocking, nb_oc_blocking;
    int ur_w, ur_w_tail;
    int typesize_in;
    int typesize_out;
    int typesize_acc;
    int typesize_conv0_bias;
    int typesize_conv1_bias;
    DataType dst_dt, conv0_bias_dt, conv1_bias_dt;
    round_mode conv0_round_mode, conv1_round_mode;
    conv_loop_order_t loop_order;
    /* conv 1x1*/
    int oc1x1;
    int oc1x1_block;
    int nb_oc1x1;
    bool use_vnni;
    bool fuse_conv1x1;
    bool conv0_with_relu;
    bool conv1_with_relu;
    bool conv0_with_bias;
    bool conv1_with_bias;
    bool conv0_multi_oc_scale;  // whether use multi channel to scale oc
    bool conv1_multi_oc_scale;
 };


struct jit_conv_call_t {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *src_prf;
    const void *dst_prf;
    const void *filt_prf;
    const void *bias_prf;
    const void *scales;
    const void *acc_s32;
    size_t kd_padding;
    size_t kd_padding_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    size_t channel;
    size_t channel_prf;
    size_t oc_blocks;
    size_t ur_w;
    size_t ur_str_w;
    size_t ch_blocks;
    int flags;
};

struct jit_conv_conf_t {
    conv_version_t ver;
    conv_loop_order_t loop_order;

    int ndims;
    int mb;
    int ngroups, ic, oc;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    bool with_bias, with_relu;
    float relu_negative_slope;
    bool with_sum;
    bool is_dw;
    int ihp, iwp, ohp, owp;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_g, g_block;
    int nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    int nb_ic_blocking_max;
    int nb_ic_L2;
    int nb_oc_L2;
    int ur_h, ur_w;
    int ur_w_tail;
    bool is_1stconv;
    /* fma avx512_core */
    conv_kernel_kind_t kernel_kind;
    /* 4fma */
    int tr_iw;
    int tr_src_num_guard_elems;
    /* 1st conv: 4fma */
    int tr_ld;
    int kh_step;
    /* 4vnni */
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;
    int tr_ow;
    /* avx512_u8s8u8 */
    int ic_nb1, ic_nb2;
    int oc_nb1;
    int ur_ow_max, ur_ow, ur_ow_tail;
    int ur_ow_nsteps;
    DataType bia_dt;
    DataType dst_dt;
    /* avx512: max possible value is nregs(32) - aux_regs(4) */
    int src_offsets[28];
    int src_count;
    bool expl_bcast;
    bool large_spatial;
    int is_oc_scale;

    // dw conv
    int nb_ch, ch_block, nb_ch_blocking;
    round_mode rm;

    // pooling
    PoolingType pool_alg;
    int pool_kw;

    //the scale for post sum
    float sum_scale;

    // output layout nhwc
    bool output_nhwc;

    int is, os, ks;
    ptrdiff_t im2col_sz;
    bool need_im2col;
    int nthr;
};

struct jit_1x1_conv_conf_t {
    conv_version_t ver;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, ow, oh;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    bool with_bias, with_relu;
    float relu_negative_slope;
    bool with_sum;

    int is, os;
    int ic_block, oc_block;

    int ur, ur_tail;

    int reduce_dim, reduce_block, nb_reduce,
        nb_reduce_blocking, nb_reduce_blocking_max;
    int load_dim, load_block, nb_load,
        nb_load_blocking, nb_load_blocking_max;
    int bcast_dim, bcast_block, nb_bcast,
        nb_bcast_blocking, nb_bcast_blocking_max;

    int reduce_loop_unroll, reduce_loop_bcast_step, reduce_loop_load_step;
    int load_loop_load_step, load_loop_iter_step;
    int bcast_loop_output_step, bcast_loop_output_substep;
    int bcast_loop_bcast_step, bcast_loop_bcast_substep;
    int fma_step;
    int load_grp_count;
    conv_1x1_loop_order_t loop_order;
    bool use_vmovntps;
    /* avx512 core */
    bool expl_bcast;
    /* 4vnni */
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;
    /* 4fma */
    bool transpose_src;
    int tr_is;
    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
    int is_oc_scale;
    DataType bia_dt;
    DataType dst_dt;
    round_mode rm;

    //the scale for post sum
    float sum_scale;
};


// pooling
struct jit_pool_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int f_pad, t_pad, l_pad;
    PoolingType alg;
    bool pad_w_is_null;
    bool simple_alg;
    DataType ind_dt;

    int c_block, c_tail, nb_c;
    int ur_c, ur_c_tail;
    int ur_w;
    int ur_w_tail;
    size_t tail[4];
    DataType src_dt;
    DataType dst_dt;
};

struct jit_pool_call_nhwc_t {
    union {
        const char *src_i8;
        const float *src_fp32;
    };
    union {
        char *dst_i8;
        float *dst_fp32;
    };
    size_t kw_range;
    size_t kh_range;
    float  idivider;
};

struct jit_pool_call_t {
    const float *src;
    const float *dst;
    const void *indices;
    const float *src_prf;
    const float *dst_prf;
    const void *indices_prf;
    size_t oh;
    size_t kd_padding;
    size_t kh_padding;
    size_t kh_padding_shift;
    size_t kd_padding_shift;
    size_t kw_padding;
    const float* init_value;
    float ker_area_h;
};

// concat with optional relu fusion
struct jit_concat_call_t {
  const unsigned char **src;
  const int  *nb_ic;
  const unsigned char *dst;
  const float *scale;
};

struct jit_concat_conf_t {
  int           bs;
  int           h, w;
  int           oc;
  int           n_inputs;
  int           typesize;
  int           *block;      // u8: 64, s32: 16
  int           bit_depth;  // 128, 256, 512 : xmm, ymm, zmm
  bool          with_relu;
  DataType      src_dt;
  DataType      dst_dt;
  float         scale_min;
  float         *scales;
  int           *nb_ic;
  int           *ic;
  int           *tail;
};

// eltsum with optional relu fusion
struct jit_eltwise_call_t {
  const void **src;
  const void *dst;
  size_t work_amount;
};

struct jit_eltwise_conf_t {
  int           n_inputs;
  DataType      dt;
  int           typesize;
  bool          with_relu;
  const float   *scales;
};


struct jit_active_call_t {
    const float *src;
    const float *for_comparison;
    const float *dst;
    size_t work_amount;
};

struct jit_active_conf_t {
  ActiveType    act_type;
  float         alpha;
  float         beta;
};

} // namespace jit
} // namespace icesword
} // namespace noobsdnn

#endif
