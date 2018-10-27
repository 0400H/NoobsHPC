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

include(cmake/thirdparty_version.cmake)
project(NBDNN C CXX)
include(cmake/msg_color.cmake)
include(cmake/utils.cmake)
include(cmake/statistic.cmake)

# ----------------------------------------------------------------------------
# section: global noobsdnn version and lib name
# ----------------------------------------------------------------------------
# global noobsdnn version 0.1.0
set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_PATCH "0")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# noobsdnn lib name and global directories
set(noobsdnn_lib_so "noobsdnn")
set(noobsdnn_lib_static "noobsdnn_static")
# set root dir of modules
set(NBDNN_ROOT ${PROJECT_SOURCE_DIR})
include_directories(${NBDNN_ROOT})

set(NBDNN_THIRD_PARTY_PATH ${NBDNN_ROOT}/third-party)
set(NBDNN_ICESWORD ${NBDNN_ROOT}/icesword)
set(NBDNN_UTILS ${NBDNN_ROOT}/utils)
set(NBDNN_UNIT_TEST ${NBDNN_ROOT}/test)
set(NBDNN_BENCHMARK ${NBDNN_ROOT}/benchmark)

# ----------------------------------------------------------------------------
# section: options for noobsdnn
# ----------------------------------------------------------------------------
#noobsdnn data float precision
noobsdnn_option(NBDNN_TYPE_FP64 "define the FP64 for data precision." NO)
noobsdnn_option(NBDNN_TYPE_FP32 "define the FP32 for data precision." YES)
noobsdnn_option(NBDNN_TYPE_FP16 "define the FP16 for data precision." NO)
noobsdnn_option(NBDNN_TYPE_INT8 "define the INT8 for data precision." NO)

#select the plantform to build
noobsdnn_option(USE_X86_PLACE "Select the build mode for X86 place." YES)

cmake_minimum_required(VERSION ${MIN_CMAKE_V} FATAL_ERROR)


if((NOT BUILD_FAT_BIN) AND (NOT BUILD_CROSS_PLANTFORM) AND USE_CUDA)
    # Select the only nvidia gpu arch you want to be built on
    set(TARGET_GPUARCH 6.1)
endif()

# build options for cuda.
noobsdnn_option(BUILD_CUBIN "BUILD with the -cubin option in Device mode" NO if USE_CUDA)
noobsdnn_option(COMPILE_PTX "Returns a list of PTX files generated from src." NO if USE_CUDA)


# common build options
noobsdnn_option(ENABLE_DEBUG "Enable DEBUG(default) mode." NO)
noobsdnn_option(ENABLE_VERBOSE_MSG "Enable verbose=1 : compile msg during make." NO)
noobsdnn_option(DISABLE_ALL_WARNINGS "Disable all the warning msg during compile." YES)
noobsdnn_option(ENABLE_NOISY_WARNINGS "Enable noisy warning msg during compile." NO if DISABLE_ALL_WARNINGS)

# using 3rd party libs
noobsdnn_option(USE_LOGGER "Build native logger components." YES)
noobsdnn_option(USE_GLOG "Build Glog components." NO if NOT USE_LOGGER)
noobsdnn_option(USE_PROTOBUF "Build Google protobuf components." YES)
noobsdnn_option(USE_OPENCV "Use static opencv libs." NO)
noobsdnn_option(USE_BOOST "Use static BOOST libs." NO)
noobsdnn_option(USE_OPENMP "Use Openmp when in android environment." YES if TARGET_ANDROID)
noobsdnn_option(USE_GTEST "Use googletest libs." NO if BUILD_WITH_UNIT_TEST)
noobsdnn_option(USE_PYTHON "Generate py wrappers." NO)
noobsdnn_option(USE_OPENCL "Use OpenCL ." NO)
noobsdnn_option(USE_GFLAGS "Build Google gflags components." NO)
noobsdnn_option(USE_MKLDNN "Use MKLDNN libs." NO)
noobsdnn_option(USE_MKLML "Use MKLML libs." YES)
noobsdnn_option(USE_XBYAK "Use XBYAK libs." YES)
noobsdnn_option(USE_OPENMP "Use Openmp." YES)
#noobsdnn_option(USE_MKL "Use MKL libs." NO)

# build components
noobsdnn_option(BUILD_WITH_UNIT_TEST "Build noobsdnn unit test components." YES)

# build examples
noobsdnn_option(BUILD_EXAMPLES "build detection and classification examples" NO)

# build target
noobsdnn_option(BUILD_SHARED "Build noobsdnn shared lib." YES)
noobsdnn_option(BUILD_STATIC "Build noobsdnn static lib." YES if NOT BUILD_SHARED)

noobsdnn_option(ENABLE_OP_TIMER "Enable op timer mode." YES)

# ----------------------------------------------------------------------------
# section: noobsdnn compiler and linker options
# ----------------------------------------------------------------------------

if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug FORCE)
else()
    set(CMAKE_BUILD_TYPE Release FORCE)
endif()

if(USE_LOGGER)
    noobsdnn_option(ENABLE_STACKTRACES "If enable local logger with stacktrace." YES)
    noobsdnn_option(SUPPORT_PTHREADS "If enable local logger with supporting pthreads. " YES)
endif()

# ----------------------------------------------------------------------------
# section:configure a header file to pass some of the CMake settings to the source
#         code
# ----------------------------------------------------------------------------
configure_file (
        "${PROJECT_SOURCE_DIR}/cmake/config/noobsdnn_config.h"
        "${PROJECT_BINARY_DIR}/noobsdnn_config.h"
)
# add the binary tree to the search path so that noobsdnn will find noobsdnn_config.h
include_directories(${PROJECT_BINARY_DIR})

# ----------------------------------------------------------------------------
# section: find modules and configure
# ----------------------------------------------------------------------------
set(NBDNN_ICESWORD_DEPENDENCIES) # set icesword dependent third-party lib targets

# check and set base compile options
include(cmake/compiler_options.cmake)
include(cmake/find_modules.cmake)

if(USE_X86_PLACE)
    set(NBDNN_TEMP_THIRD_PARTY_PATH ${NBDNN_ROOT}/third-party)
    if(USE_MKLML)
        include(cmake/external/mklml.cmake)
    endif()
    if(USE_XBYAK)
        include(cmake/external/xbyak.cmake)
    endif()
    if(USE_MKLDNN)
        include(cmake/external/mkldnn.cmake)
    endif()
endif()

# gather all the config options to noobsdnn
include(cmake/gather.cmake)


# ----------------------------------------------------------------------------
# section: build and install noobsdnn
# ----------------------------------------------------------------------------
# add source sub_directory whick holds the cmake build module

add_subdirectory(${NBDNN_ICESWORD})
add_subdirectory(${NBDNN_BENCHMARK})

if(BUILD_WITH_UNIT_TEST)
    add_subdirectory(${NBDNN_UNIT_TEST})
endif()


noobsdnn_print_statistic()


#set(executable_output_path ${PROJECT_BINARY_DIR}/unit_test)

