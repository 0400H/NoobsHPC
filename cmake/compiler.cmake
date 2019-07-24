# Copyright (c) 2018 NoobsHPC Authors, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ----------------------------------------------------------------------------
# section: set the compiler and linker options 
# ----------------------------------------------------------------------------

# generate cmake compiler comands
if(ENABLE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(ENABLE_VERBOSE_MSG)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

set(NBHPC_EXTRA_CXX_FLAGS "")

# noobshpc_add_compile_option(-v)
noobshpc_add_compile_option(-std=c++11)
noobshpc_add_compile_option(-fPIC)
noobshpc_add_compile_option(-ldl)
noobshpc_add_compile_option(-W)
noobshpc_add_compile_option(-Wall)
noobshpc_add_compile_option(-pthread)
noobshpc_add_compile_option(-Werror=return-type)
noobshpc_add_compile_option(-Werror=address)
noobshpc_add_compile_option(-Werror=sequence-point)
noobshpc_add_compile_option(-Wno-unused-variable)
noobshpc_add_compile_option(-Wformat)
noobshpc_add_compile_option(-Wmissing-declarations)
noobshpc_add_compile_option(-Winit-self)
noobshpc_add_compile_option(-Wpointer-arith)
noobshpc_add_compile_option(-Wshadow)
noobshpc_add_compile_option(-fpermissive)
noobshpc_add_compile_option(-Wsign-promo)
noobshpc_add_compile_option(-fdiagnostics-show-option)

if(ENABLE_NOISY_WARNINGS)
    noobshpc_add_compile_option(-Wcast-align)
    noobshpc_add_compile_option(-Wstrict-aliasing=2)
    noobshpc_add_compile_option(-Wundef)
    noobshpc_add_compile_option(-Wsign-compare)
else()
    noobshpc_add_compile_option(-Wno-undef)
    noobshpc_add_compile_option(-Wno-narrowing)
    noobshpc_add_compile_option(-Wno-unknown-pragmas)
    noobshpc_add_compile_option(-Wno-delete-non-virtual-dtor)
    noobshpc_add_compile_option(-Wno-comment)
    noobshpc_add_compile_option(-Wno-sign-compare)
    noobshpc_add_compile_option(-Wno-ignored-qualifiers)
    noobshpc_add_compile_option(-Wno-enum-compare)
endif()

if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug FORCE)
    noobshpc_add_compile_option(-O0) # no optmization
    noobshpc_add_compile_option(-g) # debug
    noobshpc_add_compile_option(-pg) # gprof
    noobshpc_add_compile_option(-gdwarf-2) # for old version gcc and gdb. see: http://stackoverflow.com/a/15051109/673852 
else()
    set(CMAKE_BUILD_TYPE Release FORCE)
    noobshpc_add_compile_option(-O3)
    noobshpc_add_compile_option(-DNDEBUG)
endif()

if(USE_X86_PLACE)
    noobshpc_add_compile_option(-mavx2)
    noobshpc_add_compile_option(-fopenmp)
    noobshpc_add_compile_option(-march=native)
    noobshpc_add_compile_option(-Ofast)
    noobshpc_add_compile_option(-ffast-math)
    noobshpc_add_compile_option(-Wall)
    noobshpc_add_compile_option(-Wno-comment)
    noobshpc_add_compile_option(-Wno-unused-local-typedefs)

     # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(X86_64)
        noobshpc_add_compile_option(-Wno-long-long)
    endif()
endif()

set(CMAKE_CXX_FLAGS ${NBHPC_EXTRA_CXX_FLAGS})

if(DISABLE_ALL_WARNINGS)
    noobshpc_disable_warnings(CMAKE_CXX_FLAGS)
endif()