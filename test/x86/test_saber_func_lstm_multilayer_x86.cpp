#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mkl_cblas.h"
#include "mkl_vml_functions.h"

#include "saber/core/context.h"
#include "saber/funcs/lstm.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/activation_functions.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "x86_test_common.h"
#include "test_saber_func_lstm_multilayer_x86.h"

using namespace anakin::saber;
using namespace anakin::saber::math;
using namespace std;

typedef struct _test_lstm_params {
    int mb;
    int input_size;
    int layer_size;
    ActiveType input_activation;
    ActiveType gate_activation;
    ActiveType candidate_activation;
    ActiveType cell_activation;
    bool with_peephole;
    bool with_init_hidden;
    bool skip_input;
    int num_layers;
    int num_direction;
    bool is_reverse;
} test_lstm_params;



// multiple layers Version
void compute_ref_lstm_bi_MultiLayer_fwd(std::vector<Tensor4f*> &src, std::vector<Tensor4f*> &dst, LstmParam<Tensor4f> &param) {
    SaberStatus status = SaberSuccess;

    const Tensor4f *weights = param.weight();
    const Tensor4f *bias = param.bias();
    const Tensor4f *init_hidden = param.init_hidden();
    bool is_reverse = param.is_reverse;
    int direc_num = param.num_direction;
    Tensor4f *input = src[0];
    int layer_num = param.num_layers;
    // get sequence length
    int N = input->num();
    int input_size = input->channel();
    int layer_size = dst[0]->channel() / direc_num;

    std::vector<int> seq_offset = input->get_seq_offset();
    int seq_num = seq_offset.size() - 1;
    float *ht_out_buf = (float*)zmalloc(direc_num * layer_num * N * layer_size * sizeof(float), 4096);
    float *Ct_out_buf = (float*)zmalloc(direc_num * layer_num * N * layer_size * sizeof(float), 4096);
    float *in_buf = (float*)zmalloc(direc_num * layer_num * N * 4 * layer_size * sizeof(float), 4096);
    float *hrut_buf = (float*)zmalloc(direc_num * 4 * layer_size * sizeof(float), 4096);
    float *p_buf = (float*)zmalloc(direc_num * layer_size * sizeof(float), 4096);

    // First layer is input_size * layer_size otherwise is layer_size * layer_size
    int Wx_stride_l0 = input_size * 4 * layer_size;
    int W_stride_l0 = (input_size + layer_size) * 4 * layer_size;
    if (param.skip_input) {
        Wx_stride_l0 = 0;
        W_stride_l0 = layer_size * 4 * layer_size;
    }
    int Wx_stride_ln = layer_size * 4 * layer_size;
    int W_stride_ln = (layer_size + layer_size) * 4 * layer_size;

    float* init_h = (float*)zmalloc(layer_size * sizeof(float), 4096);
    float* init_c = (float*)zmalloc(layer_size * sizeof(float), 4096);
    memset(init_h, 0, layer_size * sizeof(float));
    memset(init_c, 0, layer_size * sizeof(float));
    float* act = (float*)zmalloc(layer_size * sizeof(float), 4096);

    for (int d = 0; d < direc_num; d++) {
        float *out = ht_out_buf + d * layer_num * N * layer_size;
        float *Ct_out = Ct_out_buf + d * layer_num * N * layer_size;
        float *in = in_buf + d * layer_num * N * 4 * layer_size;
        float *hrut = hrut_buf + d * 4 * layer_size;
        float *p = p_buf + d * layer_size;

        const float *weights_data = weights->data() + d * weights->count(1, 4);
        const float *bias_data = bias->data() + d * bias->count(1, 4);

        int direction = (direc_num == 1) ? is_reverse : d;
        for (int l = 0; l < layer_num; l++) {

            const float *init_h_data = nullptr;
            const float *init_state = nullptr;
            if (param.init_hidden()) {
                init_h_data = init_hidden->data() + d * init_hidden->count(1, 4);
                init_state = init_h_data + l * init_hidden->count(2, 4);
            }

            // get h
            float *h = out + l * N * layer_size;
            // get c
            float *c = Ct_out + l * N * layer_size;

            // get Wx = [Wfx, Wix, Wcx, Wox]
            const float *Wx = (l == 0) ? weights_data : weights_data + W_stride_l0 + (l - 1) * W_stride_ln;
            // get Wch=[ Wih Wfh Wch Woh ]
            const float *Wch = (l == 0) ? Wx + Wx_stride_l0 : Wx + Wx_stride_ln;

            // get bias
            const float *b = bias_data + l * bias->count(2, 4);
            const float *peephole = nullptr;
            if (param.with_peephole) {
                peephole = b + 4 * layer_size;
            }
            // get x
            const float *x = (l == 0) ? input->data() : (out + (l - 1) * N * layer_size);
            int x_stride = (l == 0) ? input_size : layer_size;
            // get xx = x * Wx   xx=x*[Wfx, Wix, Wcx, Wox]
            float *xx = in + l * N * 4 * layer_size;
            if (param.skip_input && l == 0) {
                cblas_saxpby(4 * N* layer_size, 1, const_cast<float *>(x), 1, 0, xx, 1);
                //cause no need for Wx
                Wx = nullptr;
            }
            else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, 4 * layer_size, x_stride, 1,
                    x, x_stride, Wx, 4 * layer_size, 0, xx, 4 * layer_size);
            }
            for (int i = 0; i < seq_num; i++) {
                // do LSTM per sequence
                int seq_len = seq_offset[i + 1] - seq_offset[i];
                for (int j = 0; j < seq_len; j++) {
                    int word_idx = (direction == 1) ? (seq_len - 1 - j) : j;
                    float *ht = h + (seq_offset[i] + word_idx) * layer_size;
                    float *Ct = c + (seq_offset[i] + word_idx) * layer_size;
                    float *xxt = xx + (seq_offset[i] + word_idx) * 4 * layer_size;
                    float *ht_1 = nullptr;
                    float *Ct_1 = nullptr;
                    cblas_saxpby(4 * layer_size, 1, xxt, 1, 0, hrut, 1);
                    if (j == 0) {
                        if (param.init_hidden()) {
                            memcpy(init_h, init_state + i * layer_size, layer_size * sizeof(float));
                            memcpy(init_c, init_state + (i + seq_num)* layer_size, layer_size * sizeof(float));
                        }
                        ht_1 = init_h;
                        Ct_1 = init_c;
                    }
                    else {
                        ht_1 = (direction == 1) ? h + (seq_offset[i] + (word_idx + 1)) * layer_size
                            : h + (seq_offset[i] + (word_idx - 1)) * layer_size;

                        Ct_1 = (direction == 1) ? c + (seq_offset[i] + (word_idx + 1)) * layer_size
                            : c + (seq_offset[i] + (word_idx - 1)) * layer_size;
                    }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 4 * layer_size, layer_size, 1, ht_1, layer_size, Wch,
                        4 * layer_size, 1, hrut, 4 * layer_size);
                    if (peephole) {
                        // peephole for it
                        vsMul(layer_size, Ct_1, peephole, p);
                        cblas_saxpby(layer_size, 1, p, 1, 1, hrut, 1);
                        vsMul(layer_size, Ct_1, peephole + layer_size, p);
                        cblas_saxpby(layer_size, 1, p, 1, 1, hrut + layer_size, 1);
                    }
                    // add bias
                    cblas_saxpby(4 * layer_size, 1, b, 1, 1, hrut, 1);
                    // gate activity for it and ft, candidate activity for cct
                    activation(layer_size, hrut, hrut, param.gate_activity);
                    activation(layer_size, hrut + layer_size, hrut + layer_size, param.gate_activity);
                    activation(layer_size, hrut + 2 * layer_size, hrut + 2 * layer_size, param.candidate_activity);
                    // calc ct
                    vsMul(layer_size, hrut, hrut + 2 * layer_size, p);
                    cblas_saxpby(layer_size, 1, p, 1, 0, Ct, 1);
                    vsMul(layer_size, hrut + layer_size, Ct_1, p);
                    cblas_saxpby(layer_size, 1, p, 1, 1, Ct, 1);

                    // peephole for ot
                    if (peephole) {
                        //p=Ct*Woc
                        vsMul(layer_size, Ct, peephole + 2 * layer_size, p);
                        //Wo[ht_1,xt]+Ct*Woc
                        cblas_saxpby(layer_size, 1, p, 1, 1, hrut + 3 * layer_size, 1);
                    }
                    // get ot
                    activation(layer_size, hrut + 3 * layer_size, hrut + 3 * layer_size, param.gate_activity);
                    // calc ht
                    activation(layer_size, Ct, act, param.cell_activity);
                    // ht=tanh(Ct)*ot
                    vsMul(layer_size, hrut + 3 * layer_size, act, ht);
                }
            }

        }
        // save ht
        for (int i = 0; i < N; i++) {
            memcpy(dst[0]->mutable_data() + i * layer_size * direc_num + d * layer_size,
                out + (layer_num - 1) * N * layer_size + i * layer_size, layer_size * sizeof(float));
        }

        // save Ct
        for (int i = 0; i < N; i++) {
       memcpy(dst[1]->mutable_data() + i * layer_size * direc_num + d * layer_size,
          Ct_out + (layer_num - 1) * N * layer_size + i * layer_size, layer_size * sizeof(float));
        }
    }


    if (ht_out_buf) {
        zfree(ht_out_buf);
        ht_out_buf = nullptr;
    }
    if (Ct_out_buf) {
        zfree(Ct_out_buf);
        Ct_out_buf = nullptr;
    }

    if (in_buf) {
        zfree(in_buf);
        in_buf = nullptr;
    }

    if (hrut_buf) {
        zfree(hrut_buf);
        hrut_buf = nullptr;
    }
    if (p_buf) {
        zfree(p_buf);
        p_buf = nullptr;
    }

    if (init_h) {
        zfree(init_h);
        init_h = nullptr;
    }
    if (init_c) {
        zfree(init_c);
        init_c = nullptr;
    }
    if (act) {
        zfree(act);
        act = nullptr;
    }

    return;
}



bool lstm_test(test_lstm_params &param) {
    std::vector<Tensor4f*> inputs;

    std::vector<int> seq_offsets;
    int total_seq_len = 0;
    int offset = 0;
    for (int i = 0; i < param.mb; i++) {
        //int seq_len = 50;
        int seq_len = rand()%50 + 50;
        total_seq_len += seq_len;
        seq_offsets.push_back(offset);
        offset += seq_len;
    }
    seq_offsets.push_back(offset);

    Shape inputShape(total_seq_len, param.input_size, 1, 1);
    if (param.skip_input) {
        inputShape[1] = param.layer_size*4;
    }
    Tensor4f *i = new Tensor4f(inputShape);
    i->set_seq_offset(seq_offsets);
    inputs.push_back(i);
    fill_tensor_host_rand<Tensor4f>(*(inputs[0]));

    int W_row_l0 = param.input_size + param.layer_size;;
    int W_row_ln = (param.num_layers - 1) * param.layer_size * 2;
    if (param.skip_input) {
        W_row_l0 = param.layer_size;
        W_row_ln= (param.num_layers - 1) * param.layer_size * 2;
    }
    Shape ref_weightShape(param.num_direction, 1, W_row_l0 + W_row_ln, 4 * param.layer_size);
    Tensor4f ref_saberWeight(ref_weightShape);
    fill_tensor_host_rand(ref_saberWeight);
    int bias_num = 4;
    if (param.with_peephole) {
        bias_num = 7;
    }
    Shape ref_biasShape(param.num_direction, param.num_layers, 1, bias_num * param.layer_size);
    Tensor4f ref_saberBias(ref_biasShape);
    fill_tensor_host_rand(ref_saberBias);

    Shape ref_hiddenShape(param.num_direction, param.num_layers, param.mb*2, param.layer_size);
    Tensor4f ref_saberHidden(ref_hiddenShape);
    fill_tensor_host_rand(ref_saberHidden);

    LstmParam<Tensor4f> lstm_param(&ref_saberWeight, &ref_saberBias, param.with_init_hidden ? &ref_saberHidden : nullptr,
                                   param.input_activation, param.gate_activation, param.cell_activation,
                                   param.candidate_activation, param.with_peephole, param.skip_input,param.is_reverse,1.0,param.num_direction,param.num_layers);
    Shape outputShape(total_seq_len, param.layer_size * param.num_direction, 1, 1);

    std::vector<Tensor4f*> ref_outputs;
    Tensor4f refOutputh(outputShape);
    Tensor4f refOutputc(outputShape);
    ref_outputs.push_back(&refOutputh);
    ref_outputs.push_back(&refOutputc);

    std::vector<Tensor4f*> vender_outputs;
    Tensor4f venderOutputh(outputShape);
    Tensor4f venderOutputc(outputShape);
    vender_outputs.push_back(&venderOutputh);
    vender_outputs.push_back(&venderOutputc);

    compute_ref_lstm_bi_MultiLayer_fwd(inputs, ref_outputs, lstm_param);

    //compute saber result
    Lstm<X86, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW> venderLstm;
    Context<X86> ctx_host;
    venderLstm.init(inputs, vender_outputs, lstm_param, SPECIFY, VENDER_IMPL, ctx_host);
    venderLstm(inputs, vender_outputs, lstm_param, ctx_host);

    bool flag = compare_tensor(*ref_outputs[0], *vender_outputs[0], 1e-3);
    flag &= compare_tensor(*ref_outputs[1], *vender_outputs[1], 1e-3);
    return flag;
}

TEST(TestSaberMultiLSTMX86, test_tensor_lstm) {
    Env<X86>::env_init();

    test_lstm_params test_param[] = {
        // batch_size, input_size, layer_size, input_activation, gate_activation, candidate_activation, cell_activation, with_peephole, with_init_hidden, skip_input,num_layers,num_directions,is_reverse
        test_lstm_params{1, 20, 5, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, false, true,2,2,false},
        test_lstm_params{1, 16, 4, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, false, false,8,1,false},
        test_lstm_params{6, 1200, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false,true,false,4,2,false},
        test_lstm_params{6, 520, 130, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, false, true,2,2,false},
        test_lstm_params{1, 1200, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, true, true,1,1,true},
        test_lstm_params{6, 55, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, true,false,4,1, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false,false,2,2, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false,false,3,1, false},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true,false,3,2, false},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true,false,1,1, false},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false,false,3,2, false},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false,false,2,1, false},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true,false,3,1, false},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true,false,1,2, false},
        test_lstm_params{6, 1200, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, false,true,5,2, false},
        test_lstm_params{6, 1200, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, false,false,2,1, false},
        test_lstm_params{6, 1200, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, false, true,true,4,2, false},
        test_lstm_params{6, 1200, 300, Active_unknow, Active_sigmoid, Active_relu, Active_sigmoid, true, true,false,3,2, false},
        /*test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true, true},
        test_lstm_params{6, 55, 300, Active_tanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, false, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, false, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, false, true, true},
        test_lstm_params{6, 55, 300, Active_stanh, Active_sigmoid, Active_relu, Active_sigmoid, true, true, true},*/
    };

    for (size_t i = 0; i < ARRAY_SIZE(test_param); i++) {
        LOG(INFO) << "case " << i;
        bool ret = lstm_test(test_param[i]);

        if (ret) {
            LOG(INFO) << "Test Passed";
        }
        else {
            LOG(ERROR) << "Test Failed";
        }
    }
}

int main(int argc, const char** argv) {
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
