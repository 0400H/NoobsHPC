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

#ifndef BUFFER_H
#define BUFFER_H

#pragma once

#include "data_traits.h"

namespace noobshpc{
namespace icesword{

template<TargetType TType>
void* fast_malloc(size_t size, size_t malloc_align = 32) {
    size_t offset = sizeof(void*) + malloc_align - 1;

    char* p = nullptr;
    if (TType == X86) {
        p = static_cast<char*>(malloc(offset + size));
    } else {
        CHECK_EQ(1, 0) << "undefined target malloc";
    }

    if (!p) {
        CHECK_EQ(1, 0) << "malloc failed";
        return nullptr;
    }

    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(malloc_align - 1)));
    static_cast<void**>(r)[-1] = p;
    memset(r, 0, size);

    return r;
}

template <TargetType TType>
class Buffer {
public:
    typedef void* TPtr;

    ~Buffer() { release_memory(); }

    explicit Buffer() : _data(nullptr), _count(0) {}

    explicit Buffer(size_t size) : _data(nullptr), _count(size) {}

    explicit Buffer(TPtr data, size_t size, int id) : _count(size) { _data = data; }


    Buffer(Buffer<TType>& buf) {
        CHECK_GT(buf._count, 0) << "input buffer is empty";
        _count = buf._count;
        _data = buf._data;
    }

    Buffer& operator = (Buffer<TType>& buf) {
        this->_count = buf._count;
        this->_data = buf._data;
        return *this;
    }

    /**
     * \brief return const data pointer
    **/
    const TPtr get_data() { return _data; }

    /**
     * \brief return mutable data pointer
    **/

    TPtr get_data_mutable() { return _data; }

    /**
     * \brief return current size of memory, in size
    **/
    size_t get_count() const { return _count; }

    /**
     * \brief free old memory, alloc new memory
    **/
    Status alloc(size_t size, size_t malloc_align=32) {
        release_memory();
        _data = (void*)fast_malloc<TType>(size, malloc_align);
        _count = size;
        return S_Success;
    }

    /**
     * \brief re-alloc memory, only if hold the data, can be relloc
    **/
    Status re_alloc(size_t size, size_t malloc_align=32) {
        if (size > _count) {
            release_memory();
            _data = (void*)fast_malloc<TType>(size, malloc_align);
        }
        _count = size;
        return S_Success;
    }

    /**
     * \brief set each bytes of _data to (value) with length of (size)
    **/
    Status mem_set(int value, size_t size) {
    	if(_count != size) {
            return S_InvalidValue;
        }
        memset(_data, value, size);
    	return S_Success;
    }

    /**
     * \brief synchronously copy from other Buf
    **/
    template <TargetType TType_t>
    Status copy_from(Buffer<TType_t>& buf) {
        CHECK_GE(_count, buf.get_count());
        memcpy((char*)_data, (char*)buf.get_data(), buf.get_count());
        return S_Success;
    }

private:
    TPtr _data;
    size_t _count;

    Status release_memory() {
        if (_count > 0) {
            _count = 0;
            if (_data) {
                free(static_cast<void**>(_data)[-1]);
            }
        }
        _data = nullptr;
        return S_Success;
    }
};

} // namespace icesword
} // namespace noobshpc

#endif // BUFFER_H

