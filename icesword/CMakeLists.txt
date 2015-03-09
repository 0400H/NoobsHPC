# Copyright (c) 2018 NoobsDNN, Anakin Authors, Inc. All Rights Reserved.
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


set(NBDNN_ICESWORD_STATIC_RELAY "" )

set(NBDNN_ICESWORD_BASE_SRC "")
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD})
noobsdnn_fetch_include_recursively(${NBDNN_UTILS})
noobsdnn_fetch_include_recursively(${NBDNN_THIRD_PARTY_PATH})

# add noobsdnn_base_source files
noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/core "cpp" NBDNN_ICESWORD_BASE_SRC)
noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/funcs "cpp" NBDNN_ICESWORD_BASE_SRC)
noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/funcs/impl/ "cpp" NBDNN_ICESWORD_BASE_SRC)

if(USE_X86_PLACE)
    noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/core/impl/x86 "cpp" NBDNN_ICESWORD_BASE_SRC)
    noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/funcs/impl/x86 "cpp" NBDNN_ICESWORD_BASE_SRC)
    noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/funcs/impl/x86/kernel "cpp" NBDNN_ICESWORD_BASE_SRC)
endif()

# compile cpp objs
# add_library(NBDNN_ICESWORD_BASE_OBJS OBJECT ${NBDNN_ICESWORD_BASE_SRC})

# whole archive for static lib
if(NOT MSVC AND NOT APPLE)
    set(WHOLE_ARCHIVE_START -Wl,--whole-archive)
    set(WHOLE_ARCHIVE_END -Wl,--no-whole-archive)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # using regular Clang or AppleClang
    set(WHOLE_ARCHIVE_START -Wl,-force_load)
    set(WHOLE_ARCHIVE_END)
endif()

set(NBDNN_ICESWORD_TEMP_COMMMON_LIB ${noobsdnn_lib_so})


# add icesword library to static
if(UNIX OR APPLE)
    if (BUILD_SHARED)
        ADD_LIBRARY(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} SHARED ${NBDNN_ICESWORD_CUDA_C_SRC_OBJS} ${NBDNN_ICESWORD_BASE_SRC})
        #$<TARGET_OBJECTS:NBDNN_ICESWORD_BASE_OBJS>)
        #if(USE_MKLDNN OR USE_MKLML OR USE_XBYAK OR USE_MKL)
        if(USE_MKLDNN OR USE_MKLML OR USE_XBYAK)
            message(STATUS ${NBDNN_ICESWORD_DEPENDENCIES})
            add_dependencies(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} ${NBDNN_ICESWORD_DEPENDENCIES})
        endif()
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES VERSION ${VERSION})
        target_link_libraries(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} ${NBDNN_LINKER_LIBS})
        target_link_libraries(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} ${NBDNN_ICESWORD_STATIC_RELAY})
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LINK_FLAGS "")
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                ${NBDNN_ROOT}/output/)
    else()
        ADD_LIBRARY(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} STATIC ${NBDNN_ICESWORD_CUDA_C_SRC_OBJS} ${NBDNN_ICESWORD_BASE_SRC})
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                ${NBDNN_ROOT}/output/)
    endif ()
endif()

set(NBDNN_ICESWORD_LIB_TARGET ${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PARENT_SCOPE)
