/* Copyright (c) 2018 NoobsDNN && Anakin Authors All Rights Reserve.

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

#ifndef NBDNN_ICESWORD_OPERATOR_X86_KERNEL_CPU_ISA_H
#define NBDNN_ICESWORD_OPERATOR_X86_KERNEL_CPU_ISA_H

#pragma once

#include <type_traits>
#include <limits.h>

#define XBYAK64
#define XBYAK_NO_OP_NAMES

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR

#include "xbyak.h"
#include "xbyak_util.h"

namespace noobsdnn {
namespace icesword {
namespace jit {

#ifdef XBYAK64
    constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
        Xbyak::Operand::RBX,
        Xbyak::Operand::RBP,
        Xbyak::Operand::R12,
        Xbyak::Operand::R13,
        Xbyak::Operand::R14,
        Xbyak::Operand::R15,
    };

    static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
                              abi_param2(Xbyak::Operand::RSI),
                              abi_param3(Xbyak::Operand::RDX),
                              abi_param4(Xbyak::Operand::RCX),
                              abi_not_param1(Xbyak::Operand::RCX);
#endif

static Xbyak::util::Cpu cpu;
typedef enum {
    isa_any,
    sse42,
    avx,
    avx2,
    avx512_common,
    avx512_core,
    avx512_core_vnni,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;  // Instruction set architecture

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<sse42> {
  static constexpr int vlen_shift = 4;
  static constexpr int vlen = 16;
  static constexpr int n_vregs = 16;
};

template <>
struct cpu_isa_traits<avx2> {
  static constexpr int vlen_shift = 5;
  static constexpr int vlen = 32;
  static constexpr int n_vregs = 16;
};

template <>
struct cpu_isa_traits<avx512_common> {
  static constexpr int vlen_shift = 6;
  static constexpr int vlen = 64;
  static constexpr int n_vregs = 32;
};

template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_common> {
};

static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    using namespace Xbyak::util;

    switch (cpu_isa) {
        case sse42:
            return cpu.has(Cpu::tSSE42);
        case avx:
            return cpu.has(Cpu::tAVX);
        case avx2:
            return cpu.has(Cpu::tAVX2);
        case avx512_common:
            return cpu.has(Cpu::tAVX512F);
        case avx512_core:
            return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
                  cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
        case avx512_core_vnni:
            return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
                  cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
                  cpu.has(Cpu::tAVX512_VNNI);
        case avx512_mic:
            return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) &&
                  cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
        case avx512_mic_4ops:
            return true && mayiuse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) &&
                  cpu.has(Cpu::tAVX512_4VNNIW);
        case isa_any:
            return true;
    }
    return false;
}

} // jit
} // icesword
} // noobsdnn

#endif // NBDNN_ICESWORD_OPERATOR_X86_KERNEL_CPU_ISA_H
