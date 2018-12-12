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


#ifndef ICESWORD_FUNCS_IMPL_X86_X86_UTILS_H
#define ICESWORD_FUNCS_IMPL_X86_X86_UTILS_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "icesword/icesword.h"
#include "icesword/core/common.h"
#include "icesword/core/tensor.h"

namespace noobsdnn {
namespace icesword {

    #define UNUSED(x) ((void)x)
    #define MAYBE_UNUSED(x) UNUSED(x)


    inline void* zmalloc(size_t size, int alignment) {
        void* ptr = NULL;

        #ifdef _WIN32
                ptr = _aligned_malloc(size, alignment);
            int rc = ptr ? 0 : -1;
        #else
                int rc = ::posix_memalign(&ptr, alignment, size);
        #endif

        return (rc == 0) ? ptr : NULL;
    }

    inline void zfree(void* p) {
        #ifdef _WIN32
                _aligned_free(p);
        #else
                ::free(p);
        #endif
    }

    struct c_compatible {
        enum { default_alignment = 4096 };

        static void* operator new (size_t sz) {
            return zmalloc(sz, default_alignment);
        }

        static void* operator new (size_t sz, void* p) {
            UNUSED(sz);
            return p;
        }

        static void* operator new[](size_t sz) {
            return zmalloc(sz, default_alignment);
        }

        static void operator delete (void* p) {
            zfree(p);
        }

        static void operator delete[](void* p) {
            zfree(p);
        }
    };

    inline void weight_padding_nhwc(Tensor<X86>* input, Tensor<X86>* output) {
        Shape shape = input->shape();
        Shape shape_padding = output->shape();
        int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];
        int oc_padding = shape_padding[0], ic_padding = shape_padding[1];;

        char* output_ptr = static_cast<char*>(output->mutable_data());
        const char* input_ptr = static_cast<const char*>(input->data());

        #pragma omp parallel for collapse(4) schedule(static)
        for (int oc = 0; oc < oc_padding; ++oc) {
            for (int ic = 0; ic < ic_padding; ++ic) {
                for (int kh = 0; kh < kh_value; ++kh) {
                    for (int kw = 0; kw < kw_value; ++kw) {
                        int input_idx = oc * ic_value * kh_value * kw_value +
                                        ic * kh_value * kw_value +
                                        kh * kw_value + kw;
                        int output_idx = oc * ic_padding * kh_value * kw_value +
                                         ic * kh_value * kw_value +
                                         kh * kw_value + kw;
                        if (oc < oc_value && ic < ic_value) {
                            *(output_ptr + output_idx) = (*(input_ptr + input_idx));
                        } else {
                            *(output_ptr + output_idx) = 0;
                        }
                    }
                }
            }
        }
    }

    inline size_t datatype_size(DataType data_type) {
        switch (data_type) {
            case AK_FLOAT:
                return sizeof(float);

            case AK_INT32:
                return sizeof(int32_t);

            case AK_INT16:
                return sizeof(int16_t);

            case AK_INT8:
                return sizeof(int8_t);

            case AK_UINT8:
                return sizeof(uint8_t);

            case AK_INVALID:
            default:
                assert(!"unknown data_type");
        }

        return 0;
    }
namespace utils {
/* a bunch of std:: analogues to be compliant with any msvs version
 *
 * Rationale: msvs c++ (and even some c) headers contain special pragma that
 * injects msvs-version check into object files in order to abi-mismatches
 * during the static linking. This makes sense if e.g. std:: objects are passed
 * through between application and library, which is not the case for mkl-dnn
 * (since there is no any c++-rt dependent stuff, ideally...). */

/* SFINAE helper -- analogue to std::enable_if */

    class VectorPrint {
    public:
        template <typename Dtype>
        static void print_float(Dtype* target) {
            float* f = (float*)target;
            printf("size = %d\n", sizeof(Dtype));

            for (int i = 0; i < sizeof(Dtype) / sizeof(float); i++) {
                printf(" %f ,", f[i]);
            }

            printf("\n");
        }
    };

    template<typename T>
    inline const T& min(const T& a, const T& b) {
        return a < b ? a : b;
    }

    template<typename T>
    inline const T& max(const T& a, const T& b) {
        return a > b ? a : b;
    }

    template <typename T, typename P>
    inline bool everyone_is(T val, P item) {
        return val == item;
    }
    template <typename T, typename P, typename... Args>
    inline bool everyone_is(T val, P item, Args... item_others) {
        return val == item && everyone_is(val, item_others...);
    }

    template <typename T, typename P>
    inline bool one_of(T val, P item) {
        return val == item;
    }
    template <typename T, typename P, typename... Args>
    inline bool one_of(T val, P item, Args... item_others) {
        return val == item || one_of(val, item_others...);
    }

    template <typename... Args>
    inline bool any_null(Args... ptrs) {
        return one_of(nullptr, ptrs...);
    }

    template<typename T>
    inline void array_copy(T* dst, const T* src, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i];
        }
    }

    template<typename T>
    inline bool array_cmp(const T* a1, const T* a2, size_t size) {
        for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) {
                return false;
            }

        return true;
    }

    template<typename T, typename U>
    inline void array_set(T* arr, const U& val, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            arr[i] = static_cast<T>(val);
        }
    }


    #ifdef USE_OPENMP
        static inline int omp_get_max_threads() {
            return 1;
        }
        static inline int omp_get_num_threads() {
            return 1;
        }
        static inline int omp_get_thread_num() {
            return 0;
        }
        static inline int omp_in_parallel() {
            return 0;
        }
    #endif //openmp

    template <typename T> struct remove_reference {
        typedef T type;
    };
    template <typename T> struct remove_reference<T&> {
        typedef T type;
    };
    template <typename T> struct remove_reference < T&& > {
        typedef T type;
    };

    template <typename T>
    inline T&& forward(typename remove_reference<T>::type& t) {
        return static_cast < T && >(t);
    }
    template <typename T>
    inline T&& forward(typename remove_reference<T>::type&& t) {
        return static_cast < T && >(t);
    }

    template <typename T>
    inline typename remove_reference<T>::type zero() {
        auto zero = typename remove_reference<T>::type();
        return zero;
    }

    template <typename T, typename U>
    inline typename remove_reference<T>::type div_up(const T a, const U b) {
        assert(b);
        return (a + b - 1) / b;
    }

    template <typename T, typename U>
    inline typename remove_reference<T>::type rnd_up(const T a, const U b) {
        return div_up(a, b) * b;
    }

    template <typename T, typename U>
    inline typename remove_reference<T>::type rnd_dn(const T a, const U b) {
        return (a / b) * b;
    }

    template <typename T, typename U>
    inline void balance211(T n, U team, U tid, T& n_start, T& n_end) {
        T n_min = 1;
        T& n_my = n_end;

        if (team <= 1 || n == 0) {
            n_start = 0;
            n_my = n;
        } else if (n_min == 1) {
            // team = T1 + T2
            // n = T1*n1 + T2*n2  (n1 - n2 = 1)
            T n1 = div_up(n, (T)team);
            T n2 = n1 - 1;
            T T1 = n - n2 * (T)team;
            n_my = (T)tid < T1 ? n1 : n2;
            n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
        }

        n_end += n_start;
    }

    template<typename T>
    inline T nd_iterator_init(T start) {
        return start;
    }
    template<typename T, typename U, typename W, typename... Args>
    inline T nd_iterator_init(T start, U& x, const W& X, Args&& ... tuple) {
        start = nd_iterator_init(start, utils::forward<Args>(tuple)...);
        x = start % X;
        return start / X;
    }

    inline bool nd_iterator_step() {
        return true;
    }
    template<typename U, typename W, typename... Args>
    inline bool nd_iterator_step(U& x, const W& X, Args&& ... tuple) {
        if (nd_iterator_step(utils::forward<Args>(tuple)...)) {
            x = (x + 1) % X;
            return x == 0;
        }

        return false;
    }

    template <typename T0, typename T1, typename F>
    inline void parallel_nd(const T0 D0, const T1 D1, F f) {
        const size_t work_amount = (size_t)D0 * D1;
        if (work_amount == 0) return;

        #pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            size_t start{0}, end{0};
            balance211(work_amount, nthr, ithr, start, end);
            T0 d0{0}; T1 d1{0};
            nd_iterator_init(start, d0, D0, d1, D1);
            for (size_t iwork = start; iwork < end; ++iwork) {
                f(d0, d1);
                nd_iterator_step(d0, D0, d1, D1);
            }
        }
    }

    template <typename T0, typename T1, typename T2, typename F>
    inline void parallel_nd(const T0 D0, const T1 D1, const T2 D2, F f) {
        const size_t work_amount = (size_t)D0 * D1 * D2;
        if (work_amount == 0) return;

        #pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            size_t start{0}, end{0};
            balance211(work_amount, nthr, ithr, start, end);
            T0 d0{0}; T1 d1{0}; T2 d2{0};
            nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
            for (size_t iwork = start; iwork < end; ++iwork) {
                f(d0, d1, d2);
                nd_iterator_step(d0, D0, d1, D1, d2, D2);
            }
        }
    }

    template<typename U, typename W, typename Y>
    inline bool nd_iterator_jump(U& cur, const U end, W& x, const Y& X) {
        U max_jump = end - cur;
        U dim_jump = X - x;

        if (dim_jump <= max_jump) {
            x = 0;
            cur += dim_jump;
            return true;
        } else {
            cur += max_jump;
            x += max_jump;
            return false;
        }
    }

    template<typename U, typename W, typename Y, typename... Args>
    inline bool nd_iterator_jump(U& cur, const U end, W& x, const Y& X,
                                 Args&& ... tuple) {
        if (nd_iterator_jump(cur, end, utils::forward<Args>(tuple)...)) {
            x = (x + 1) % X;
            return x == 0;
        }

        return false;
    }

} // namespace utils
} // namespace icesword
} // namespace noobsdnn

#endif // X86_UTILS_H
