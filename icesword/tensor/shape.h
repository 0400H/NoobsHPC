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

#ifndef NBDNN_ICESWORD_TENSOR_SHAPE_H
#define NBDNN_ICESWORD_TENSOR_SHAPE_H

#include <vector>

#include "icesword/tensor/layout.h"
#include "utils/logger/logger.h"

namespace noobsdnn{
namespace icesword{

class Shape : public std::vector<int> {
public:
    using vector = std::vector<int>;

    ~Shape() {
        delete _layout;
        _layout = nullptr;
    }

    Shape() : vector(), _layout(nullptr) {}
    Shape(vector data, LayoutType LT_TYPE = LT_NCHW) {
        create_layout(LT_TYPE);
        CHECK_EQ(_layout->dims(), data.size());
        for (int i = 0; i < _layout->dims(); ++i) {
            this->push_back(data[i]);
        }
    }
    Shape(const Shape& right)
        : std::vector<int>(right) {
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
        create_layout(right.get_layout());
    }

    Shape &operator=(const Shape& right) {
        this->clear();
        for (int i = 0; i < right.size(); ++i) {
            this->push_back(right[i]);
        }
        delete _layout;
        _layout = nullptr;
        create_layout(right.get_layout());
        return *this;
    }
    Shape operator+(const Shape& shape) {
        Shape tmp_shape(*this);
        int* p = data();
        for (size_t i = 0; i < size(); i++) {
            tmp_shape[i] = p[i] + shape[i];
        }
        return tmp_shape;
    }
    Shape operator-(const Shape& shape) {
        Shape tmp_shape(*this);
        int* p = data();
        for (size_t i = 0; i < size(); i++) {
            tmp_shape[i] = p[i] - shape[i];
        }
        return tmp_shape;
    }
    bool operator<(const Shape& shape) const {
        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i < size(); i++) {
            flag = flag && (p[i] < shape[i]);
        }
        return flag;
    }
    bool operator<=(const Shape& shape) const{
        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i < size(); i++) {
            flag = flag && (p[i] <= shape[i]);
        }
        return flag;
    }
    bool operator>(const Shape& shape) const {
        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i > size(); i++) {
            flag = flag && (p[i] > shape[i]);
        }
        return flag;
    }
    bool operator>=(const Shape& shape) const{
        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i > size(); i++) {
            flag = flag && (p[i] >= shape[i]);
        }
        return flag;
    }
    bool operator==(const Shape& shape) const{
        bool flag = size() == shape.size();
        if (!flag) {
            return false;
        }
        const int* p = data();
        for (size_t i = 0; i < size(); i++) {
            flag = flag && (p[i] == shape[i]);
        }
        return flag;
    }

    int group_index() const {
        if (_layout) {
            return _layout->group_index();
        } else {
            return -1;
        }
    }
    int batch_index() const {
        if (_layout) {
            return _layout->batch_index();
        } else {
            return -1;
        }
    }
    int channel_index() const {
        if (_layout) {
            return _layout->channel_index();
        } else {
            return -1;
        }
    }
    int height_index() const {
        if (_layout) {
            return _layout->height_index();
        } else {
            return -1;
        }
    }
    int width_index() const {
        if (_layout) {
            return _layout->width_index();
        } else {
            return -1;
        }
    }
    int depth_index() const {
        if (_layout) {
            return _layout->depth_index();
        } else {
            return -1;
        }
    }

    int dims() const {
        return this->size();
    }
    int group() const {
        int shape_batch = this->group_index() == -1 ? 1 : this->data()[this->group_index()];
        return shape_batch;
    }
    int batch() const {
        int shape_batch = this->batch_index() == -1 ? 1 : this->data()[this->batch_index()];
        return shape_batch;
    }
    int channel() const {
        int shape_channel = this->channel_index() == -1 ? 1 : this->data()[this->channel_index()];
        return shape_channel;
    }
    int height() const {
        int shape_height = this->height_index() == -1 ? 1 : this->data()[this->height_index()];
        return shape_height;
    }
    int width() const {
        int shape_width = this->width_index() == -1 ? 1 : this->data()[this->width_index()];
        return shape_width;
    }
    int depth() const {
        int shape_depth = this->depth_index() == -1 ? 1 : this->data()[this->depth_index()];
        return shape_depth;
    }

    long long count(int start = 0) const {
        if (start > dims()) {
            start = dims();
        }
        if (this->size() == 0) {
            return 0;
        }
        long long sum = 1;
        for_each(this->begin() + start, this->end(), [&](int n){sum *= n;});
        return sum;
    }
    long long count(int start, int end) const {
        if (start < 0) {
            start = 0;
        }
        if (end > dims()) {
            end = dims();
        }
        if (end < start) {
            end = start;
        }
        long long  sum  = 1;
        for (int i = start; i < end; ++i) {
            sum *= data()[i];
        }
        return sum;
    }

    Shape get_stride() const {
        Shape data_stride = Shape::zero(*this);
        for (int i = 0; i < dims(); ++i) {
            data_stride[i] = count(i + 1);
        }
        return data_stride;
    }

    LayoutType get_layout() const {
        if (_layout) {
            return _layout->get_layouttype();
        } else {
            return LT_invalid;
        }
    }

    void set_group (const int group) {
        CHECK_GT(group, 0);
        if (_layout->group_index() != -1) {
            this->data()[_layout->group_index()] = group;
        }
    }
    void set_batch (const int batch) {
        CHECK_GT(batch, 0);
        if (_layout->batch_index() != -1) {
            this->data()[_layout->batch_index()] = batch;
        }
    }
    void set_channel (const int channel) {
        CHECK_GT(channel, 0);
        if (_layout->channel_index() != -1) {
            this->data()[_layout->channel_index()] = channel;
        }
    }
    void set_height (const int height) {
        CHECK_GT(height, 0);
        if (_layout->height_index() != -1) {
            this->data()[_layout->height_index()] = height;
        }
    }
    void set_width (const int width) {
        CHECK_GT(width, 0);
        if (_layout->width_index() != -1) {
            this->data()[_layout->width_index()] = width;
        }
    }
    void set_depth (const int depth) {
        CHECK_GT(depth, 0);
        if (_layout->depth_index() != -1) {
            this->data()[_layout->depth_index()] = depth;
        }
    }
    void set_layout(LayoutType LT_TYPE, std::vector<int> new_shape = {}) {
        Shape sh = *this;
        Layout* layout = this->_layout;
        create_layout(LT_TYPE);
        if (sh._layout == nullptr) {
            return;
        }
        this->clear();
        if (new_shape.size() != 0) {
            CHECK_EQ(_layout->dims(), new_shape.size()) << "new_shape dims miss match with layout dims";
            for (auto i : new_shape) {
                this->push_back(i);
            }
            return;
        }
        this->resize(_layout->dims());
        if (_layout->group_index() != -1) {
            this->data()[_layout->group_index()] = sh.group();
        }
        if (_layout->batch_index() != -1) {
            this->data()[_layout->batch_index()] = sh.batch();
        }
        if (_layout->channel_index() != -1) {
            this->data()[_layout->channel_index()] = sh.channel();
        }
        if (_layout->height_index() != -1) {
            this->data()[_layout->height_index()] = sh.height();
        }
        if (_layout->width_index() != -1) {
            this->data()[_layout->width_index()] = sh.width();
        }
        if (_layout->depth_index() != -1) {
            this->data()[_layout->depth_index()] = sh.depth();
        }
        delete layout;
    }
    static Shape zero(const Shape &right){
        Shape sh = right;
        for (int i = 0; i < right.size(); ++i) {
            sh[i] = 0;
        }
        return sh;
    }

    static Shape minusone(const Shape &right){
        Shape sh = right;
        for (int i = 0; i < right.size(); ++i) {
            sh[i] = -1;
        }
        return sh;
    }

protected:
    Layout* _layout{nullptr};
private:
    void create_layout(LayoutType LT_TYPE) {
        switch(LT_TYPE) {
            case LT_invalid: this->_layout = nullptr; break;
            case LT_C: this->_layout = new C(); break;
            case LT_NC: this->_layout = new NC(); break;
            case LT_NGC: this->_layout = new NGC(); break;
            case LT_HW: this->_layout = new HW(); break;
            case LT_WH: this->_layout = new WH(); break;
            case LT_NHW: this->_layout = new NHW(); break;
            case LT_NWH: this->_layout = new NWH(); break;
            case LT_NCHW: this->_layout = new NCHW(); break;
            case LT_NHWC: this->_layout = new NHWC(); break;
            case LT_NCHW_C8: this->_layout = new NCHW_C8(); break;
            case LT_NCHW_C16: this->_layout = new NCHW_C16(); break;
            case LT_NGCHW: this->_layout = new NGCHW(); break;
            case LT_NGHWC: this->_layout = new NGHWC(); break;
            case LT_NHWGC: this->_layout = new NHWGC(); break;
            case LT_GNHWC: this->_layout = new GNHWC(); break;
            case LT_GNCHW: this->_layout = new GNCHW(); break;
            default : LOG(FATAL) << "don't support layout";
        }
    }
};

} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_TENSOR_SHAPE_H
