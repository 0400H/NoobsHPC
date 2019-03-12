#===============================================================================
# Copyright 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

include(ExternalProject)

set(MKLDNN_PROJECT        "extern_mkldnn")
set(MKLDNN_SOURCE_DIR     ${NBDNN_THIRD_PARTY_PATH}/mkldnn)
set(MKLDNN_INSTALL_DIR    ${MKLDNN_SOURCE_DIR})
set(MKLDNN_INCLUDE_DIR    "${MKLDNN_INSTALL_DIR}/include" CACHE PATH "mkldnn include directory." FORCE)
set(MKLDNN_LIB            "${MKLDNN_INSTALL_DIR}/lib/libmkldnn.so" CACHE FILEPATH "mkldnn library." FORCE)

include_directories(${MKLDNN_INCLUDE_DIR})

set(MKLDNN_DEPENDS ${MKLML_PROJECT})

message(STATUS "Scanning external modules ${Green}MKLDNNN${ColourReset}...")

if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS "5.4")
    set(MKLDNN_CFLAG)
else()
   set(MKLDNN_CFLAG "${CMAKE_C_FLAGS} -Wno-error=strict-overflow \
   -Wno-unused-but-set-variable -Wno-unused-variable -Wno-format-truncation")
endif()

if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "5.4")
  set(MKLDNN_CXXFLAG)
else()
    set(MKLDNN_CXXFLAG "${CMAKE_CXX_FLAGS} -Wno-error=strict-overflow \
   -Wno-unused-but-set-variable -Wno-unused-variable -Wno-format-truncation")
endif()

set(MKLDNN_C_COMPILER ${CMAKE_C_COMPILER})
set(MKLDNN_CXX_COMPILER ${CMAKE_CXX_COMPILER})
ExternalProject_Add(
    ${MKLDNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ${MKLDNN_DEPENDS}
    GIT_REPOSITORY      "https://github.com/01org/mkl-dnn.git"
    GIT_TAG             "a7c5f53832acabade6e5086e72c960adedb3c38a"
    PREFIX              ${MKLDNN_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR}
    CMAKE_ARGS          -DMKLROOT=${MKLML_INSTALL_ROOT}
    CMAKE_ARGS          -DCMAKE_C_COMPILER=${MKLDNN_C_COMPILER}
    CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${MKLDNN_CXX_COMPILER}
    CMAKE_ARGS          -DCMAKE_C_FLAGS=${MKLDNN_CFLAG}
    CMAKE_ARGS          -DCMAKE_CXX_FLAGS=${MKLDNN_CXXFLAG}
    CMAKE_ARGS          -DWITH_TEST=OFF -DWITH_EXAMPLE=OFF
)

# check and download
add_library(mkldnn SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB})
add_dependencies(mkldnn ${MKLDNN_PROJECT})

list(APPEND NBDNN_ICESWORD_DEPENDENCIES mkldnn)
list(APPEND NBDNN_LINKER_LIBS ${MKLDNN_LIB})