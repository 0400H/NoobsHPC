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

#ifndef NBDNN_ICESWORD_LAYOUT_H
#define NBDNN_ICESWORD_LAYOUT_H

#include "icesword/types.h"

namespace noobsdnn{
namespace icesword{

struct Layout {
    virtual int batch_index() {return -1;}
    virtual int group_index() {return -1;}
    virtual int channel_index() {return -1;}
    virtual int height_index() {return -1;}
    virtual int width_index() {return -1;}
    virtual int depth_index() {return -1;}
    virtual int dims() {return -1;}
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
struct NGC : public Layout {
    int batch_index() {return 0;}
    int group_index() {return 1;}
    int channel_index() {return 2;}
    int dims() {return 3;}
    LayoutType get_layouttype() {return LT_NGC;}
};
struct HW : public Layout {
    int height_index() {return 0;}
    int width_index() {return 1;}
    int dims() {return 2;}
    LayoutType get_layouttype() {return LT_HW;}
};
struct WH : public Layout {
    int height_index() {return 1;}
    int width_index() {return 0;}
    int dims() {return 2;}
    LayoutType get_layouttype() {return LT_WH;}
};
struct NHW : public Layout {
    int batch_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int dims() {return 3;}
    LayoutType get_layouttype() {return LT_NHW;}
};
struct NWH : public Layout {
    int batch_index() {return 0;}
    int width_index() {return 1;}
    int height_index() {return 2;}
    int dims() {return 3;}
    LayoutType get_layouttype() {return LT_NWH;}
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
struct NCHW_C8 : public Layout {
    int batch_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 8;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_NCHW_C8;}
};
struct NCHW_C16 : public Layout {
    int batch_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int inner_c() {return 16;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_NCHW_C16;}
};
struct GCHW : public Layout {
    int group_index() {return 0;}
    int channel_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int dims() {return 4;}
    LayoutType get_layouttype() {return LT_GCHW;}
};
struct GHWC : public Layout {
    int group_index() {return 0;}
    int height_index() {return 1;}
    int width_index() {return 2;}
    int channel_index() {return 3;}
    int dims() {return 4;}
    LayoutType get_layouttype() {return LT_GHWC;}
};
struct NGCHW : public Layout {
    int batch_index() {return 0;}
    int group_index() {return 1;}
    int channel_index() {return 2;}
    int height_index() {return 3;}
    int width_index() {return 4;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_NGCHW;}
};
struct NGHWC : public Layout {
    int batch_index() {return 0;}
    int group_index() {return 1;}
    int height_index() {return 3;}
    int width_index() {return 4;}
    int channel_index() {return 2;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_NGHWC;}
};
struct NHWGC : public Layout {
    int batch_index() {return 0;}
    int group_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int channel_index() {return 4;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_NHWGC;}
};
struct GNCHW : public Layout {
    int group_index() {return 0;}
    int batch_index() {return 1;}
    int height_index() {return 2;}
    int width_index() {return 3;}
    int channel_index() {return 4;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_GNCHW;}
};
struct GNHWC : public Layout {
    int group_index() {return 0;}
    int batch_index() {return 1;}
    int height_index() {return 3;}
    int width_index() {return 4;}
    int channel_index() {return 2;}
    int dims() {return 5;}
    LayoutType get_layouttype() {return LT_GNHWC;}
};


} // namespace icesword
} // namespace noobsdnn

#endif // NBDNN_ICESWORD_LAYOUT_H