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

#ifndef LAYOUT_H
#define LAYOUT_H

#pragma once

#include "icesword/types.h"

namespace noobshpc{
namespace icesword{

struct Layout {
    virtual int dims() {return -1;}
    virtual int batch_index() {return -1;}
    virtual int group_index() {return -1;}
    virtual int height_index() {return -1;}
    virtual int width_index() {return -1;}
    virtual int depth_index() {return -1;}
    virtual int channel_index() {return -1;}
    virtual int in_channel_index() {return -1;}
    virtual int out_channel_index() {return -1;}
    virtual LayoutType get_layouttype() {return LT_invalid;}
};

struct C : public Layout {
    int channel_index() {return 0;}
    int dims() {return 1;}
    LayoutType get_layouttype() {return LT_C;}
};
struct NC : public Layout {
    int batch_index() {return 0;}
    int channel_index() {return 1;}
    int dims() {return 2;}
    LayoutType get_layouttype() {return LT_NC;}
};
struct HW : public Layout {
    int height_index() {return 0;}
    int width_index() {return 1;}
    int dims() {return 2;}
    LayoutType get_layouttype() {return LT_HW;}
};
struct NGC : public Layout {
    int batch_index() {return 0;}
    int group_index() {return 1;}
    int channel_index() {return 2;}
    int dims() {return 3;}
    LayoutType get_layouttype() {return LT_NGC;}
};
struct NCHW : public Layout {
    int batch_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int dims() {return 4;}
    LayoutType get_layouttype() {return LT_NCHW;}
};
struct NHWC : public Layout {
    int batch_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int channel_index() {return 3;}
    int dims() {return 4;}
    LayoutType get_layouttype() {return LT_NHWC;}
};
struct GOHWI : public Layout {
    int group_index() {return 0;}
    int out_channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int in_channel_index() {return 4;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_GOHWI;}
};
struct GOIHW : public Layout {
    int group_index() {return 0;}
    int out_channel_index() {return 1;}
    int in_channel_index() {return 2;}
    int height_index() {return 3;}
    int width_index() {return 4;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_GOIHW;}
};

} // namespace icesword
} // namespace noobshpc

#endif // LAYOUT_H