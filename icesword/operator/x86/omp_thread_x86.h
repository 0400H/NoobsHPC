/*  Copyright (c) 2018 NoobsDNN Authors All Rights Reserve.

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


#ifndef ICESWORD_OPERATOR_X86_OMP_THREAD_H
#define ICESWORD_OPERATOR_X86_OMP_THREAD_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string>
#include <iostream>

#include "mkl.h"
#include "icesword/operator/operator.h"
#include "icesword/core/tensor/tensor_op.h"

namespace noobsdnn {
namespace icesword {

#ifdef USE_OPENMP
    #include <omp.h>
    inline int ice_get_max_threads() { return omp_get_max_threads(); }
    inline int ice_get_num_threads() { return omp_get_num_threads(); }
    inline int ice_get_thread_num() { return omp_get_thread_num(); }
    inline int ice_in_parallel() { return omp_in_parallel(); }
#endif

/* general parallelization */
template <typename F>
void parallel(int nthr, F func) {
    if (nthr == 0) nthr = ice_get_max_threads();
    if (nthr == 1) { func(0, 1); return; }
    #pragma omp parallel num_threads(nthr)
    func(ice_get_thread_num(), ice_get_num_threads());
}

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
static inline T&& forward(typename remove_reference<T>::type& t) {
    return static_cast < T && >(t);
}
template <typename T>
static inline T&& forward(typename remove_reference<T>::type&& t) {
    return static_cast < T && >(t);
}

template <typename T>
static inline typename remove_reference<T>::type zero() {
    auto zero = typename remove_reference<T>::type();
    return zero;
}

template <typename T, typename U>
static inline typename remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
static inline typename remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename U>
static inline typename remove_reference<T>::type rnd_dn(const T a, const U b) {
    return (a / b) * b;
}

template <typename T, typename U>
static inline void balance211(T n, U team, U tid, T& n_start, T& n_end) {
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
static inline T nd_iterator_init(T start) {
    return start;
}
template<typename T, typename U, typename W, typename... Args>
static inline T nd_iterator_init(T start, U& x, const W& X, Args&& ... tuple) {
    start = nd_iterator_init(start, forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

static inline bool nd_iterator_step() {
    return true;
}
template<typename U, typename W, typename... Args>
static inline bool nd_iterator_step(U& x, const W& X, Args&& ... tuple) {
    if (nd_iterator_step(forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }

    return false;
}

template <typename T0, typename T1, typename F>
static inline void parallel_nd(const T0 D0, const T1 D1, F f) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;

    #pragma omp parallel
    {
        const int ithr = ice_get_thread_num();
        const int nthr = ice_get_num_threads();
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
static inline void parallel_nd(const T0 D0, const T1 D1, const T2 D2, F f) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;

    #pragma omp parallel
    {
        const int ithr = ice_get_thread_num();
        const int nthr = ice_get_num_threads();
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
static inline bool nd_iterator_jump(U& cur, const U end, W& x, const Y& X) {
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
static inline bool nd_iterator_jump(U& cur, const U end, W& x, const Y& X,
                                Args&& ... tuple) {
    if (nd_iterator_jump(cur, end, forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }

    return false;
}

} // namespace icesword
} // namespace noobsdnn

#endif // ICESWORD_OPERATOR_X86_OMP_THREAD_H
