/* Copyright (c) 2018 NoobsDNN, Anakin Authors, Inc. All Rights Reserved.

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

#ifndef NBDNN_ICESWORD_CORE_TARGET_WRAPPER_H
#define NBDNN_ICESWORD_CORE_TARGET_WRAPPER_H

#include <memory>

#include "icesword/core/target_traits.h"
#include "icesword/core/data_traits.h"

namespace noobsdnn{

namespace icesword {

const int MALLOC_ALIGN = 64;


static inline void* fast_malloc(size_t size) {
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p = static_cast<char*>(malloc(offset + size));

    if (!p) {
        return nullptr;
    }

    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    memset(r, 0, size);
    return r;
}

static inline void fast_free(void* ptr) {
    if (ptr) {
        free(static_cast<void**>(ptr)[-1]);
    }
}

template <TargetType>
class Buffer;

template <TargetType TType,
          typename target_category = typename TargetTypeTraits<TType>::target_category>
struct TargetWrapper {};

/**
 * \brief for host target only, arm, x86
 * only support host to host memory copy
 *
*/
template <TargetType TType>
struct TargetWrapper<TType, __host_target> {
    typedef void* event_t;
    typedef void* stream_t;

    /**
     * \brief get total device number of target type
     * @param count
     */
    static void get_device_count(int& count) {
        // todo
        LOG(WARNING) << "host target \" get_device_count\" is not implemented";
        count = 1;
    }

    static void set_device(int id) {
        // todo
    }

    /**
     * \brief wrapper of memory allocate function, with alignment of 16 bytes
     *
    */
    static void mem_alloc(void** ptr, size_t n) {
        *ptr = (void*)fast_malloc(n);

    }

    /**
     * \brief wrapper of memory free function
     *
    */
    static void mem_free(void* ptr) {
        if (ptr != nullptr) {
            fast_free(ptr);
        }
    }

    /**
     * \brief wrapper of memory set function, input value only supports 0 or -1
     *
    */
    static void mem_set(void* ptr, int value, size_t n) {
        memset(ptr, value, n);
    }

    /**
     * \brief create event, empty function for host target
     *
    */
    static void create_event(event_t* event, bool flag = false) {}

    /**
     * \brief destroy event, empty function for host target
     *
    */
    static void destroy_event(event_t event) {}

    /**
     * \brief create stream, empty function for host target
     *
    */
    static void create_stream(stream_t* stream) {}

    /**
     * \brief create stream with flag, empty function for host target
     *
    */
    static void create_stream_with_flag(stream_t* stream, unsigned int flag) {}


    /**
     * \brief create stream with priority, empty function for host target
     *
    */
    static void create_stream_with_priority(stream_t* stream, unsigned int flag, int priority) {}

    /**
     * \brief destroy event, empty function for host target
     *
    */
    static void destroy_stream(stream_t stream) {}

    /**
     * \brief record event, empty function for host target
     *
    */
    static void record_event(event_t event, stream_t stream) {}

    /**
     * \brief query event, empty function for host target
     *
    */
    static void query_event(event_t event) {}

    /**
     * \brief synchronize event, empty function for host target
     *
    */
    static void sync_event(event_t event) {}

    /**
     * \brief crreate event, empty function for host target
     *
    */
    static void sync_stream(event_t event, stream_t stream) {}

    static void sync_stream(stream_t stream) {}

    /**
     * \brief memory copy function, use memcopy from host to host
     *
    */
    static void sync_memcpy(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count, __HtoH) {
        memcpy((char*)dst + dst_offset, (char*)src + src_offset, count);
        //LOG(INFO) << "host, sync, H2H, size: " << count;
    }

    /**
     * \brief same with sync_memcpy
     * @tparam void
     * @param dst
     * @param dst_id
     * @param src
     * @param src_id
     * @param count
     */
    static void async_memcpy(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, \
        size_t count, stream_t stream, __HtoH) {
        memcpy((char*)dst + dst_offset, (char*)src + src_offset, count);
        //LOG(INFO) << "host, sync, H2H, size: " << count;
    }

    /**
     * \brief memcpy peer to peer, for device memory copy between different devices
     * @tparam void
     * @param dst
     * @param dst_dev
     * @param src
     * @param src_dev
     * @param count
     */
    static void sync_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, size_t count) {}

    /**
     * \brief asynchronize memcpy peer to peer, for device memory copy between different devices
     * @tparam void
     * @param dst
     * @param dst_dev
     * @param src
     * @param src_dev
     * @param count
     */
    static void async_memcpy_p2p(void* dst, size_t dst_offset, int dst_id, \
        const void* src, size_t src_offset, int src_id, size_t count, stream_t stream) {}

    /**
     * \brief host target return 0
     * @return      always return 0
     */
    static int get_device_id() {
        return 0;
    }

    static void device_sync() {}
};

} //namespace icesword

} //namespace noobsdnn

#endif //NBDNN_ICESWORD_CORE_TARGET_WRAPPER_H
