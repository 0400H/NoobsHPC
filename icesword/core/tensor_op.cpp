#include "tensor_op.h"

#include <cstdlib>
#include <random>

namespace noobsdnn {

namespace icesword {

template <class Tensor_t>
void fill_tensor_host_const(Tensor_t& tensor, typename Tensor_t::Dtype value) {

    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        data_ptr[i] = value;
    }
}

template <class Tensor_t>
void fill_tensor_host_rand(Tensor_t& tensor) {
    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());

    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>((float)rand() / RAND_MAX - 0.5);
        // data_ptr[i] = static_cast<Dtype>((float)rand() - RAND_MAX/2);
        // data_ptr[i] = static_cast<Dtype>((float)rand());
    }
}

template <class Tensor_t>
void fill_tensor_host_seq(Tensor_t& tensor) {
    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());

    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>(i);
    }
}

template <class Tensor_t>
void fill_tensor_host_rand(Tensor_t& tensor, typename Tensor_t::Dtype vstart, \
                           typename Tensor_t::Dtype vend) {
    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        Dtype random_num = vstart + (vend - vstart) * dis(gen);
        data_ptr[i] = random_num;
    }
}

template <class Tensor_t>
void print_tensor_host(Tensor_t& tensor) {

    typedef typename Tensor_t::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    const Dtype* data_ptr = static_cast<const Dtype*>(tensor.get_buf()->get_data());
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        printf("%.2f ", static_cast<float>(data_ptr[i]));

        if ((i + 1) % tensor.width() == 0) {
            printf("\n");
        }
    }

    printf("\n");
}

template <typename Dtype>
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, \
                     int size, double& max_ratio, double& max_diff) {

    const double eps = 1e-6f;
    max_diff = fabs(src1[0] - src2[0]);
    max_ratio = 2.0 * max_diff / (src1[0] + src2[0] + eps);

    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1[i] - src2[i]);

        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (src1[i] + src2[i] + eps);
        }
    }
}

#define FILL_TENSOR_HOST(target, type, layout) \
    template void fill_tensor_host_const<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor, DataTrait<target, type>::dtype value); \
    template void fill_tensor_host_rand<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor); \
    template void fill_tensor_host_rand<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor, DataTrait<target, type>::dtype vstart, \
        DataTrait<target, type>::dtype vend); \
    template void print_tensor_host<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor);\
    template void fill_tensor_host_seq<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor);


FILL_TENSOR_HOST(X86, AK_FLOAT, NCHW);
FILL_TENSOR_HOST(X86, AK_FLOAT, NCHW_C8);
FILL_TENSOR_HOST(X86, AK_FLOAT, NHWC);
FILL_TENSOR_HOST(X86, AK_FLOAT, NHW);
FILL_TENSOR_HOST(X86, AK_FLOAT, NW);
FILL_TENSOR_HOST(X86, AK_FLOAT, HW);
FILL_TENSOR_HOST(X86, AK_FLOAT, W);

FILL_TENSOR_HOST(X86, AK_INT32, NCHW);
FILL_TENSOR_HOST(X86, AK_INT32, NHWC);
FILL_TENSOR_HOST(X86, AK_INT32, NHW);
FILL_TENSOR_HOST(X86, AK_INT32, NW);
FILL_TENSOR_HOST(X86, AK_INT32, HW);
FILL_TENSOR_HOST(X86, AK_INT32, W);

FILL_TENSOR_HOST(X86, AK_INT8, NCHW);
FILL_TENSOR_HOST(X86, AK_INT8, NHWC);
FILL_TENSOR_HOST(X86, AK_INT8, NHW);
FILL_TENSOR_HOST(X86, AK_INT8, NW);
FILL_TENSOR_HOST(X86, AK_INT8, HW);
FILL_TENSOR_HOST(X86, AK_INT8, W);

FILL_TENSOR_HOST(X86, AK_UINT8, NCHW);
FILL_TENSOR_HOST(X86, AK_UINT8, NHWC);
FILL_TENSOR_HOST(X86, AK_UINT8, NHW);
FILL_TENSOR_HOST(X86, AK_UINT8, NW);
FILL_TENSOR_HOST(X86, AK_UINT8, HW);
FILL_TENSOR_HOST(X86, AK_UINT8, W);

template void tensor_cmp_host<float>(const float* src1, const float* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<char>(const char* src1, const char* src2, int size, \
                                    double& max_ratio, double& max_diff);

template <>
void print_tensor_host<Tensor<X86, AK_INT8, NCHW_C4>>(Tensor<X86, AK_INT8, NCHW_C4>& tensor) {
    typedef typename Tensor<X86, AK_INT8, NCHW_C4>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    const Dtype* data_ptr = tensor.data();
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        printf("%.2f ", static_cast<float>(data_ptr[i]));

        if ((i + 1) % (4 * tensor.width()) == 0) {
            printf("\n");
        }
    }

    printf("\n");
}

#ifdef USE_X86_PLACE

template <>
void print_tensor_host<Tensor<X86, AK_FLOAT, NCHW_C16>>(Tensor<X86, AK_FLOAT, NCHW_C16>& tensor) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C16>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    const Dtype* data_ptr = tensor.data();
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        printf("%.2f ", data_ptr[i]);

        if ((i + 1) % (16 * tensor.width()) == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <>
void fill_tensor_host_u8_rand<Tensor<X86, AK_FLOAT, NHWC>>(Tensor<X86, AK_FLOAT, NHWC> & tensor) {}

template <>
void fill_tensor_host_u8_rand<Tensor<X86, AK_FLOAT, NCHW_C16>>(Tensor<X86, AK_FLOAT, NCHW_C16> & tensor) {}

template <>
void fill_tensor_host_u8_rand<Tensor<X86, AK_UINT8, NHWC>>(Tensor<X86, AK_UINT8, NHWC> & tensor) {
    typedef typename Tensor<X86, AK_UINT8, NHWC>::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>((int)rand()%64);
    }
}

template <>
void fill_tensor_host_s8_rand<Tensor<X86, AK_INT8, NCHW>>(Tensor<X86, AK_INT8, NCHW> & tensor) {
    typedef typename Tensor<X86, AK_INT8, NCHW>::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>((int)rand()%127 - 64);
    }
}

template <>
void fill_tensor_host_s8_rand<Tensor<X86, AK_INT8, NHWC>>(Tensor<X86, AK_INT8, NHWC> & tensor) {
    typedef typename Tensor<X86, AK_INT8, NHWC>::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>((int)rand()%127 - 64);
    }
}

template <>
void reorder<Tensor<X86, AK_UINT8, NCHW>, Tensor<X86, AK_UINT8, NHWC>>(Tensor<X86, AK_UINT8, NCHW>& src, Tensor<X86, AK_UINT8, NHWC>& dst) {
    typedef typename Tensor<X86, AK_UINT8, NHWC>::Dtype outDtype;
    typedef typename Tensor<X86, AK_UINT8, NCHW>::Dtype inDtype;
    const inDtype *src_data = src.data();
    outDtype *dst_data = dst.mutable_data();
    int width = src.width();
    int height = src.height();
    int channel = src.channel();
    const int spatial_size = height * width;
    auto ker = [&](const inDtype *i, outDtype *o) {
        for (int c = 0; c < channel; ++c) {
            const size_t nchw_off = c * spatial_size;
            o[c] = i[nchw_off];
        }
    };
    int num = dst.num();
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int input_offset =  (n * channel * height + h) * width + w;
                int output_offset = ((n * height + h) * width + w) * channel;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}

template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW>, Tensor<X86, AK_UINT8, NCHW>>(Tensor<X86, AK_FLOAT, NCHW>& src, Tensor<X86, AK_UINT8, NCHW>& dst) {
    typedef typename Tensor<X86, AK_UINT8, NHWC>::Dtype outDtype;
    typedef typename Tensor<X86, AK_FLOAT, NCHW>::Dtype inDtype;
    const inDtype *src_data = src.data();
    outDtype *dst_data = dst.mutable_data();
    int width = src.width();
    int height = src.height();
    int channel = src.channel();
    auto ker = [&](const inDtype *i, outDtype *o) {
        for (int w = 0; w < width; ++w) {
            const size_t nchw_off = w;
            o[w] = nearbyintf(i[nchw_off]);
        }
    };
    int num = dst.num();
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < height; ++h) {
                int input_offset =  ((n * channel + c) * height + h) * width;
                int output_offset = ((n * channel + c) * height + h) * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}
template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW>, Tensor<X86, AK_FLOAT, NCHW_C16>>(Tensor<X86, AK_FLOAT, NCHW>& src, Tensor<X86, AK_FLOAT, NCHW_C16>& dst) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C16>::Dtype Dtype;
    int blksize = 16;
    const Dtype *src_data = src.data();
    Dtype *dst_data = dst.mutable_data();
    int width = src.width();
    int height = src.height();
    const int spatial_size = height * width;
    auto ker = [&](const Dtype *i, Dtype *o) {
        for (int w = 0; w < src.width(); ++w) {
            for (int c = 0; c < blksize; ++c) {
                const size_t nchw_off = c * spatial_size + w;
                o[w * blksize + c] = i[nchw_off];
            }
        }
    };
    int num = dst.num();
    int channel = src.channel();
    int channel_blk = channel / blksize;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int C = 0; C < channel_blk; ++C) {
            for (int h = 0; h < height; ++h) {
                int input_offset = ((n * channel + blksize * C) * height + h) * width;
                int output_offset = ((n * channel_blk + C) * height + h) * blksize * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}

template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW_C16>, Tensor<X86, AK_FLOAT, NCHW>>(Tensor<X86, AK_FLOAT, NCHW_C16>& src, Tensor<X86, AK_FLOAT, NCHW>& dst) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C16>::Dtype Dtype;
    int blksize = 16;
    const Dtype *src_data = src.data();
    Dtype *dst_data = dst.mutable_data();
    int width = dst.width();
    int height = dst.height();
    const int spatial_size = height * width;
    auto ker = [&](const Dtype *i, Dtype *o) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < blksize; ++c) {
                const size_t nchw_off = c * spatial_size + w;
                o[nchw_off] = i[w * blksize + c];
            }
        }
    };
    int num = dst.num();
    int channel = dst.channel();
    int channel_blk = channel / blksize;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int C = 0; C < channel_blk; ++C) {
            for (int h = 0; h < height; ++h) {
                int input_offset = ((n * channel_blk + C) * height + h) * blksize * width;
                int output_offset = ((n * channel + blksize * C) * height + h) * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}
template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW_C8>, Tensor<X86, AK_FLOAT, NCHW>>(Tensor<X86, AK_FLOAT, NCHW_C8>& src, Tensor<X86, AK_FLOAT, NCHW>& dst) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C8>::Dtype Dtype;
    int blksize = 8;
    const Dtype *src_data = src.data();
    Dtype *dst_data = dst.mutable_data();
    int width = dst.width();
    int height = dst.height();
    const int spatial_size = height * width;
    auto ker = [&](const Dtype *i, Dtype *o) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < blksize; ++c) {
                const size_t nchw_off = c * spatial_size + w;
                o[nchw_off] = i[w * blksize + c];
            }
        }
    };
    int num = dst.num();
    int channel = dst.channel();
    int channel_blk = channel / blksize;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int C = 0; C < channel_blk; ++C) {
            for (int h = 0; h < height; ++h) {
                int input_offset = ((n * channel_blk + C) * height + h) * blksize * width;
                int output_offset = ((n * channel + blksize * C) * height + h) * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}
#endif

#ifdef USE_CUDA

template<>
SaberStatus
DataTensorTransformHelper::convert_weights<Tensor<X86, AK_INT8, NCHW_C4>,
                          Tensor<X86, AK_FLOAT, NCHW> >(Tensor<X86, AK_INT8, NCHW_C4>& out_tensor,
                                  const Tensor<X86, AK_FLOAT, NCHW>& in_tensor,
Context<NV> ctx) {
    int input_channel = in_tensor.channel();
    int output_channel = out_tensor.shape()[1];
    //            LOG(INFO)<<"input_channel = "<<input_channel<<" output_channel = "<<output_channel;
    _vector_weight_scale.resize(input_channel);

    int weight_inner_dim = in_tensor.channel()
                           * in_tensor.height()
                           * in_tensor.width();
    const float* in_weight_data = in_tensor.data();

    for (int c = 0; c < input_channel; ++c) {
        float max_val = -1.f;

        for (int i = 0; i < weight_inner_dim; ++i) {
            float read_data = fabs(in_weight_data[i]);
            max_val = (read_data > max_val) ? read_data : max_val;
        }

        _vector_weight_scale[c] = max_val / 127.f;
        in_weight_data += weight_inner_dim;
        //                LOG(INFO)<<"max_val = "<<max_val<<" vector: "<<max_val / 127.f;
    }

    int o_num = out_tensor.num();
    int o_channel = output_channel;
    int o_height = out_tensor.height();
    int o_width = out_tensor.width();

    int out_n_stride = o_channel * o_height * o_width;
    int out_c_stride = o_height * o_width;
    int out_h_stride = o_width;

    Shape in_stride = in_tensor.get_stride();

    in_weight_data = in_tensor.data();
    char* out_weight_data = out_tensor.mutable_data();

    for (int idx = 0; idx < o_num * o_channel * o_height * o_width; ++idx) {

        int n = (idx / (out_n_stride)) % o_num;
        int in_offset = ((idx / (out_n_stride)) % o_num) * in_stride[0]
                        + ((idx / (out_c_stride)) % o_channel) * (in_stride[1] * 4)
                        + ((idx / (out_h_stride)) % o_height) * in_stride[2]
                        + (idx % o_width) * in_stride[3];

        int out_offset = ((idx / (out_n_stride)) % o_num) * out_n_stride
                         + ((idx / (out_c_stride)) % o_channel) * out_c_stride
                         + ((idx / (out_h_stride)) % o_height) * out_h_stride
                         + (idx % o_width);
        out_weight_data[out_offset * 4 + 0] = (char)(round(
                in_weight_data[in_offset + 0 * in_stride[1]] / _vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 1] = (char)(round(
                in_weight_data[in_offset + 1 * in_stride[1]] / _vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 2] = (char)(round(
                in_weight_data[in_offset + 2 * in_stride[1]] / _vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 3] = (char)(round(
                in_weight_data[in_offset + 3 * in_stride[1]] / _vector_weight_scale[n]));

    }

    return SaberSuccess;
}
template<>
SaberStatus
DataTensorTransformHelper::convert_bias<Tensor<X86, AK_FLOAT, NCHW>,
                          Tensor<X86, AK_FLOAT, NCHW> >(Tensor<X86, AK_FLOAT, NCHW>& out_tensor,
                                  const Tensor<X86, AK_FLOAT, NCHW>& in_tensor,
Context<NV> ctx) {
    unsigned long weight_size = _vector_weight_scale.size();
    unsigned long bias_size = in_tensor.size();
    CHECK_GT(_in_scale, 0);
    CHECK_GT(weight_size, 0);
    CHECK_EQ(bias_size, weight_size);

    const float* in_data = in_tensor.data();
    float* out_data = out_tensor.mutable_data();

    for (int i = 0; i < bias_size; ++i) {
        out_data[i] = in_data[i] / _in_scale / _vector_weight_scale[i];
    }

    return SaberSuccess;
}
#endif

} //namespace icesword

} //namespace noobsdnn
