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

set(MIN_CMAKE_V 3.0)
cmake_minimum_required(VERSION ${MIN_CMAKE_V} FATAL_ERROR)

project(NBHPC C CXX)

include(cmake/colors.cmake)
include(cmake/utils.cmake)

# ----------------------------------------------------------------------------
# section: global noobshpc version and lib name
# ----------------------------------------------------------------------------
# global noobshpc version 0.0.1

set(VERSION_MAJOR "0")
set(VERSION_MINOR "0")
set(VERSION_PATCH "1")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# noobshpc lib name and global directories
set(noobshpc_lib_so "noobshpc")
set(noobshpc_lib_static "noobshpc_static")

# set root dir of modules
set(NBHPC_ROOT ${PROJECT_SOURCE_DIR})
set(NBHPC_CMAKE ${NBHPC_ROOT}/cmake)
set(NBHPC_UTILS ${NBHPC_ROOT}/utils)
set(NBHPC_INCLUDE ${NBHPC_ROOT}/include)
set(NBHPC_ICESWORD ${NBHPC_ROOT}/icesword)
set(NBHPC_UNIT_TEST ${NBHPC_ROOT}/unit_test)
set(NBHPC_BENCHMARK ${NBHPC_ROOT}/benchmark)
set(NBHPC_THIRD_PARTY_PATH ${NBHPC_ROOT}/third-party)

# set build and output dir
set(NBHPC_BUILD ${NBHPC_ROOT}/build)
set(LIBRARY_OUTPUT_DIRECTORY ${NBHPC_BUILD}/lib)
set(RUNTIME_OUTPUT_DIRECTORY ${NBHPC_BUILD}/output)

# ----------------------------------------------------------------------------
# section: options for noobshpc
# ----------------------------------------------------------------------------

# common build options
noobshpc_option(ENABLE_DEBUG "Enable DEBUG(default) mode." YES)
noobshpc_option(ENABLE_VERBOSE_MSG "Enable verbose=1 : compile msg during make." NO)
noobshpc_option(ENABLE_EXPORT_COMPILE_COMMANDS "Export compile command to json." YES)
noobshpc_option(DISABLE_ALL_WARNINGS "Disable all the warning msg during compile." YES)
noobshpc_option(ENABLE_NOISY_WARNINGS "Enable noisy warning msg during compile." NO if DISABLE_ALL_WARNINGS)


# noobshpc data precision
noobshpc_option(NBHPC_TYPE_FP64 "define the FP64 for data precision." NO)
noobshpc_option(NBHPC_TYPE_FP32 "define the FP32 for data precision." YES)
noobshpc_option(NBHPC_TYPE_FP16 "define the FP16 for data precision." NO)
noobshpc_option(NBHPC_TYPE_INT8 "define the INT8 for data precision." YES)

# using common library
noobshpc_option(USE_OPENMP "Use Openmp." YES)
noobshpc_option(USE_LOGGER "Build baidu logger components." YES)

# select the plantform to build
noobshpc_option(USE_X86_PLACE "Select the build mode for X86 place." YES)

# build components
noobshpc_option(BUILD_WITH_UNIT_TEST "Build noobshpc unit_test components." YES)
noobshpc_option(BUILD_WITH_BENCHMARK "Build noobshpc benchmark components." NO)

# build library target
noobshpc_option(BUILD_SHARED "Build noobshpc shared lib." YES)
noobshpc_option(BUILD_STATIC "Build noobshpc static lib." YES if NOT BUILD_SHARED)

# x86 rely on 3rd party libs
if(USE_X86_PLACE)
    noobshpc_option(USE_MKLDNN "Use MKLDNN libs." NO)
    noobshpc_option(USE_MKLML "Use MKLML libs." YES)
    noobshpc_option(USE_XBYAK "Use XBYAK libs." YES)
endif()

# ----------------------------------------------------------------------------
# section: find modules and configure
# ----------------------------------------------------------------------------

if(USE_MKLML)
    include(${NBHPC_CMAKE}/external/mklml.cmake)
endif()
if(USE_XBYAK)
    include(${NBHPC_CMAKE}/external/xbyak.cmake)
endif()
if(USE_MKLDNN)
    include(${NBHPC_CMAKE}/external/mkldnn.cmake)
endif()

# ----------------------------------------------------------------------------
# section: noobshpc compiler and linker options
# ----------------------------------------------------------------------------

include(${NBHPC_CMAKE}/compiler.cmake)

# ----------------------------------------------------------------------------
# section: add dir to header files search path
# ----------------------------------------------------------------------------

# add the binary tree to the search path so that noobshpc will find noobshpc_config.h
configure_file ("${PROJECT_SOURCE_DIR}/cmake/config/noobshpc.config"
                "${NBHPC_BUILD}/noobshpc_config.h")

include_directories(${NBHPC_ROOT})
include_directories(${NBHPC_INCLUDE})
include_directories(${PROJECT_BINARY_DIR})

# ----------------------------------------------------------------------------
# section: add dir to source files search path
# ----------------------------------------------------------------------------

add_subdirectory(${NBHPC_ICESWORD})

if(BUILD_WITH_UNIT_TEST)
    add_subdirectory(${NBHPC_UNIT_TEST})
endif()

if(BUILD_WITH_BENCHMARK)
    add_subdirectory(${NBHPC_BENCHMARK})
endif()

# ----------------------------------------------------------------------------
# section: show status
# ----------------------------------------------------------------------------

include(cmake/status.cmake)

# ----------------------------------------------------------------------------
# section: set make doc flag
# ----------------------------------------------------------------------------
include(cmake/doc.cmake)