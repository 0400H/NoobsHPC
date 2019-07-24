/* Copyright (c) 2018 NoobsHPC Authors, Inc. All Rights Reserved.

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

#ifndef TENSOR_H
#define TENSOR_H

#pragma once

#include "shape.h"
#include "buffer.h"

namespace noobshpc{
namespace icesword{

template<TargetType TType>
class Tensor {
public:
    typedef void* BaseDtype;

    Tensor(DataType type = DT_FLOAT) : _valid_shape(), _shape(), _offset() {
        _dtype = type;
        _type_len = type_length(type);
        _buf = std::make_shared<Buffer<TType>>();
        _is_subbuf = false;
    }

    Tensor(Shape shape, DataType type = DT_FLOAT) {
        _shape = shape;
        _valid_shape = shape;
        _offset = Shape::zero(shape);
        _dtype = type;
        _type_len = type_length(type);
        _buf = std::make_shared<Buffer<TType>>(shape.count() * _type_len);
        _is_shared = false;
        _is_subbuf = false;
    }

    Tensor(const Tensor<TType>& tensor) {
        _shape = tensor._shape;
        _valid_shape = tensor._valid_shape;
        _offset = tensor._offset;
        _dtype = tensor._dtype;
        _type_len = tensor._type_len;
        _buf = tensor._buf;
        _is_subbuf = tensor._is_subbuf;
        _is_shared = tensor._is_shared;
        _seq_offset = tensor._seq_offset;
        _scale = tensor._scale;
    }

    /**
     * \brief set scale for different precision data convert
     * @param scale
    **/
    void set_scale(std::vector<float> scale) { _scale = scale; }

    /**
     * \brief get scale
     * @param scale
    **/
    std::vector<float> get_scale() const { return _scale; }

    Status set_dtype(DataType type) {
        _dtype = type;
        _type_len = type_length(type);
        if (_buf->get_count() < _shape.count() * _type_len) {
            _buf->re_alloc(_shape.count() * _type_len);
        }
        return S_Success;
    }

    /**
     * \brief get tensor's DataType, DT_INT8 / DT_FLOAT ...
     * @return
    **/
    DataType get_dtype() const { return _dtype; }

    /**
     * \brief change tensor's layout and type
     * @param layout
     * @param data
     * @return
    **/
    Status set_layout(LayoutType layout, std::vector<int> data = {}) {
        _valid_shape.set_layout(layout, data);
        return S_Success;
    }

    /**
     * \brief get tensor's Layout, DT_INT8 / DT_FLOAT ...
     * @return
    **/
    LayoutType get_layout() const { return _shape.get_layout(); }

    /**
     *  \brief Free old buffer and alloc a new tensor buffer.
    **/
    Status re_alloc(Shape shape) {
        CHECK_EQ(_is_shared || _is_subbuf, false) << "shared tensor could not re_alloc";
        _type_len = type_length(_dtype);
        _shape = shape;
        _valid_shape = _shape;
        _offset =Shape::zero(_shape);
        _buf->alloc(_shape.count() * _type_len);
        return S_Success;
    }

    Status re_alloc(Shape shape, DataType type, size_t malloc_align = 32) {
        CHECK_EQ(_is_shared || _is_subbuf, false) << "shared tensor could not re_alloc";
        _dtype = type;
        _type_len = type_length(type);
        _shape = shape;
        _valid_shape = _shape;
        _offset =Shape::zero(_shape);
        _buf->alloc(_shape.count() * _type_len, malloc_align);
        return S_Success;
    }

    /**
     *  \brief only change the shape and valid shape, do nothing to memory
     *  \param shape
     *  \param valid_shape
     *  \param offset
    **/
    Status set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {
        if (shape.dims() > 0) {
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            _offset = offset;
        }
        CHECK_EQ(valid_shape > Shape::zero(valid_shape), true) << "valid_shape size should > 0";
        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() <= _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(valid_shape);
        } else {
            if (_shape == Shape::zero(_valid_shape)) {
                _shape = valid_shape;
            }
            //if (!(_valid_shape + _offset <= _shape)) { \
                return S_InvalidValue; \
            }
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        return S_Success;
    }

    /**
     *  \brief Change tensor shape,
     *  if input shape's count is bigger than the capacity of buffer, alloc a new buffer.
    **/
    Status reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {
        if (shape.dims() > 0) {
            _shape = shape;
        }
        if (offset.dims() > 0 && _is_subbuf) {
            _offset = offset;
        }
        CHECK_EQ(valid_shape > Shape::zero(valid_shape), true) << \
            "valid_shape size should > 0";
        _valid_shape = valid_shape;

        if (!_is_subbuf) {
            if (_shape.count() < _valid_shape.count()) {
                _shape = _valid_shape;
            }
            _offset = Shape::zero(_valid_shape);
        } else {
            if (_shape == Shape::zero(valid_shape)) {
                _shape = valid_shape;
            }
            //if (!(_valid_shape + _offset <= _shape)) { \
                return S_InvalidValue; \
            }
            CHECK_EQ(_valid_shape + _offset <= _shape, true) << \
                "valid_shape + offet should <= shape";
        }
        bool exceed_flag = _shape.count() * _type_len > _buf->get_count() \
            && (_is_subbuf || _is_shared);
        CHECK_EQ(exceed_flag, false) << "shared tensor shape exceed origin data buffer size";
        ICESWORD_CHECK(_buf->re_alloc(_shape.count() * _type_len));
        return S_Success;
    }

    // bool is_continue_mem() const {
    //     if (!_is_subbuf) {
    //         return true;
    //     }
    //     return _valid_shape.is_continue(_shape);
    // }

    /**
     *  \brief Return shape count, from start index to end index(end index is excluded).
     *  \param start Input start index.
     *  \param end   Input end index (exclude in calculation).
     *  \return the size from start index to end index.
    **/
    long long count(int start, int end) const { return _shape.count(start, end); }

    /**
     *  \brief return valid_shape count, from start index to end index(end index is excluded).
     *  \param start input start index.
     *  \param end   input end index (exclude in calculation).
     *  \return the size from start index to end index.
    **/
    long long count_valid(int start, int end) const { return _valid_shape.count(start, end); }

    /**
     *  \brief Return tensor shape size, not the valid shape size.
    **/
    long long size() const { return _shape.count(); }

    /**
     *  \brief Return the valid shape size.
     *  \return Return the valid shape size.
    **/
    long long valid_size() const{ return _valid_shape.count(); }

    /**
     *  \brief Return tensor shape dims.
    **/
    int dims() const { return _valid_shape.dims(); }

    /**
     *  \brief Return tensor shape, entire memory buffer shape.
    **/
    Shape shape() const{ return _shape; }

    /**
     *  \brief Return valid shape of tensor
    **/
    Shape valid_shape() const { return _valid_shape; }

    /**
     *  \brief compute data stride.
    **/
    Shape get_stride() const {
        if (_is_subbuf) {
            return  _shape.get_stride();
        }
        return  _valid_shape.get_stride();
    }

    /**
     *  \brief Return tensor offset, which holds the offset in each dim.
    **/
    Shape offset() const { return _offset; }

    /**
     *  \brief Return valid shape of tensor
    **/
    int data_offset() const { return start_index(); }


    /**
     * \brief get sequence offset, lot tensor
     * @return
    **/
    std::vector<std::vector<int>> get_seq_offset() const { return _seq_offset; }

    /**
     * \brief set sequence offset, lot tensor
     * @param seq_offset
     * @return
    **/
    Status set_seq_offset(std::vector<std::vector<int>> seq_offset) {
        _seq_offset = seq_offset;
        return S_Success;
    }

    /**
     *  \brief Return number
    **/
    int batch() const { return _valid_shape.batch(); }

    /**
     *  \brief Return number index in shape.
    **/
    int batch_index() const { return _valid_shape.batch_index(); }

    /**
     *  \brief set number to valid shape.
    **/
    void set_batch(int num) {
        _valid_shape.set_batch(num);
        if (_shape.count() < _valid_shape.count()) {
            _shape = _valid_shape;
        }
    }

    /**
     *  \brief Return channel.
    **/
    int channel() const { return _valid_shape.channel(); }

    /**
     *  \brief Return channel index in shape.
     *  \return
    **/
    int channel_index() const { return _valid_shape.channel_index(); }

    /**
     *  \brief set channel to valid shape.
    **/
    void set_channel(int channel) {
        _valid_shape.set_channel(channel);
        if (_shape.count() < _valid_shape.count()) {
            _shape = _valid_shape;
        }
    }

    /**
     *  \brief Return height.
     *  \return
    **/
    int height() const { return _valid_shape.height(); }

    /**
     *  \brief Return height index in shape.
     *  \return
    **/
    int height_index() const { return _valid_shape.height_index(); }

    /**
     *  \brief set height to valid shape.
    **/
    void set_height(int h) {
        _valid_shape.set_height(h);
        if (_shape.count() < _valid_shape.count()) {
            _shape = _valid_shape;
        }
    }

    /**
     *  \brief Return width.
     *  \return
    **/
    int width() const { return _valid_shape.width(); }

    /**
     *  \brief Return height index in shape.
     *  \return
    **/
    int width_index() const { return _valid_shape.width_index(); }

    /**
     *  \brief set width to valid shape.
    **/
    void set_width(int w) {
        _valid_shape.set_width(w);
        if (_shape.count() < _valid_shape.count()) {
            _shape = _valid_shape;
        }
    }

    /**
     *  \brief Return tensor mutable data pointer void*.
    **/
    BaseDtype mutable_data() {
        if (_buf->get_count() == 0){
            return nullptr;
        }
        return static_cast<BaseDtype>(_buf->get_data_mutable());
    }

    /**
     *  \brief Return tensor data pointer, with data type of current tensor (Dtype*).
    **/
    const BaseDtype data() const {
        if (_buf->get_count() == 0){
            return nullptr;
        }
        return static_cast<const BaseDtype>(_buf->get_data());
    }

    /**
     *  \brief Deep copy data within region of interest from input tensor.
    **/
    template <TargetType TType_t>
    Status copy_from(const Tensor<TType_t>& tensor) {
        CHECK_EQ(tensor.get_dtype(), _dtype) << "data type should be the same";
        CHECK_EQ(valid_size(), tensor.valid_size()) \
            << "sizes of two valid shapes must be the same";

        if (_buf->get_count() == 0) {
            reshape(_valid_shape);
        }

        /// get the proper process target wrapper
        typedef typename DataTraitBase<TType_t>::PtrDtype BaseDtype_src;

        int dst_data_offset = data_offset();
        int src_data_offset = tensor.data_offset();

        BaseDtype ptr_dst = _buf->get_data_mutable();
        const BaseDtype_src ptr_src = tensor.data();

        memcpy((char*)ptr_dst + _type_len * dst_data_offset,
                (char*)ptr_src + _type_len * src_data_offset,
                _type_len * valid_size());

        return S_Success;
    }

private:
    //! scale for quantization
    std::vector<float> _scale;

    ///< Length of datatype.
    DataType _dtype;
    size_t _type_len;

    ///< Represent the raw mem shape.
    Shape _shape;
    ///< Represent the mem you have right to access shape.
    Shape _valid_shape;
    ///< Represent the offset idx between _shape and _real_shape.
    Shape _offset;
    ///< Buffer shared ptr, hold the data pointer, and buffer capacity.
    std::shared_ptr<Buffer<TType>> _buf{nullptr};
    ///< share sub-buffer flag.
    bool _is_subbuf{false};
    bool _is_shared{false};

    //! lot tensor
    std::vector<std::vector<int>> _seq_offset;

    /// Get data real start index.
    int start_index() const {
        if (!_is_subbuf) {
            return 0;
        }
        Shape stride = get_stride();
        int idx = 0;
        for (int i = 0; i < stride.size(); ++i) {
            idx += _offset[i] * stride[i];
        }
        return idx;
    }
};


} // namespace icesword
} // namespace noobshpc

#endif // TENSOR_H