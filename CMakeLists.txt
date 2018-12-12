# Copyright (c) 2018 NoobsDNN Authors, Inc. All Rights Reserved.
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

project(NBDNN C CXX)

include(cmake/colors.cmake)
include(cmake/utils.cmake)

# ----------------------------------------------------------------------------
# section: global noobsdnn version and lib name
# ----------------------------------------------------------------------------
# global noobsdnn version 0.0.1

set(VERSION_MAJOR "0")
set(VERSION_MINOR "0")
set(VERSION_PATCH "1")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# noobsdnn lib name and global directories
set(noobsdnn_lib_so "noobsdnn")
set(noobsdnn_lib_static "noobsdnn_static")

# set root dir of modules
set(NBDNN_ROOT ${PROJECT_SOURCE_DIR})
set(NBDNN_CMAKE ${NBDNN_ROOT}/cmake)
set(NBDNN_INCLUDE ${NBDNN_ROOT}/include)
set(NBDNN_ICESWORD ${NBDNN_ROOT}/icesword)
set(NBDNN_UTILS ${NBDNN_ROOT}/utils)
set(NBDNN_UNIT_TEST ${NBDNN_ROOT}/test)
set(NBDNN_BENCHMARK ${NBDNN_ROOT}/benchmark)
set(NBDNN_THIRD_PARTY_PATH ${NBDNN_ROOT}/third-party)

# set build and output dir
set(NBDNN_BUILD ${NBDNN_ROOT}/build)
set(LIBRARY_OUTPUT_DIRECTORY ${NBDNN_BUILD})
set(RUNTIME_OUTPUT_DIRECTORY ${NBDNN_BUILD}/output)

# ----------------------------------------------------------------------------
# section: options for noobsdnn
# ----------------------------------------------------------------------------

# common build options
noobsdnn_option(ENABLE_DEBUG "Enable DEBUG(default) mode." NO)
noobsdnn_option(ENABLE_VERBOSE_MSG "Enable verbose=1 : compile msg during make." NO)
noobsdnn_option(DISABLE_ALL_WARNINGS "Disable all the warning msg during compile." YES)
noobsdnn_option(ENABLE_NOISY_WARNINGS "Enable noisy warning msg during compile." NO if DISABLE_ALL_WARNINGS)

# noobsdnn data precision
noobsdnn_option(NBDNN_TYPE_FP64 "define the FP64 for data precision." NO)
noobsdnn_option(NBDNN_TYPE_FP32 "define the FP32 for data precision." YES)
noobsdnn_option(NBDNN_TYPE_FP16 "define the FP16 for data precision." NO)
noobsdnn_option(NBDNN_TYPE_INT8 "define the INT8 for data precision." YES)

# using common library
noobsdnn_option(USE_OPENMP "Use Openmp." YES)
noobsdnn_option(USE_LOGGER "Build baidu logger components." YES)

# select the plantform to build
noobsdnn_option(USE_X86_PLACE "Select the build mode for X86 place." YES)

# build components
noobsdnn_option(BUILD_WITH_UNIT_TEST "Build noobsdnn unit_test components." YES)
noobsdnn_option(BUILD_WITH_BENCHMARK "Build noobsdnn benchmark components." NO)

# build library target
noobsdnn_option(BUILD_SHARED "Build noobsdnn shared lib." YES)
noobsdnn_option(BUILD_STATIC "Build noobsdnn static lib." YES if NOT BUILD_SHARED)

# x86 rely on 3rd party libs
if(USE_X86_PLACE)
    noobsdnn_option(USE_MKLDNN "Use MKLDNN libs." NO)
    noobsdnn_option(USE_MKLML "Use MKLML libs." YES)
    noobsdnn_option(USE_XBYAK "Use XBYAK libs." YES)
endif()

# ----------------------------------------------------------------------------
# section: find modules and configure
# ----------------------------------------------------------------------------

if(USE_MKLML)
    include(${NBDNN_CMAKE}/external/mklml.cmake)
endif()
if(USE_XBYAK)
    include(${NBDNN_CMAKE}/external/xbyak.cmake)
endif()
if(USE_MKLDNN)
    include(${NBDNN_CMAKE}/external/mkldnn.cmake)
endif()

# ----------------------------------------------------------------------------
# section: noobsdnn compiler and linker options
# ----------------------------------------------------------------------------

include(${NBDNN_CMAKE}/compiler.cmake)

# ----------------------------------------------------------------------------
# section: add dir to header files search path
# ----------------------------------------------------------------------------

# add the binary tree to the search path so that noobsdnn will find noobsdnn_config.h
configure_file ("${PROJECT_SOURCE_DIR}/cmake/config/noobsdnn.config"
                "${PROJECT_BINARY_DIR}/noobsdnn_config.h")

include_directories(${NBDNN_ROOT})
include_directories(${NBDNN_INCLUDE})
include_directories(${PROJECT_BINARY_DIR})

# ----------------------------------------------------------------------------
# section: add dir to source files search path
# ----------------------------------------------------------------------------

add_subdirectory(${NBDNN_ICESWORD})

if(BUILD_WITH_UNIT_TEST)
    add_subdirectory(${NBDNN_UNIT_TEST})
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(${NBDNN_BENCHMARK})
endif()

# ----------------------------------------------------------------------------
# section: show status
# ----------------------------------------------------------------------------

include(cmake/status.cmake)

# ----------------------------------------------------------------------------
# section: set make doc flag
# ----------------------------------------------------------------------------
include(cmake/doc.cmake)