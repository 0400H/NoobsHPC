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

set(NBDNN_ICESWORD_STATIC_RELAY "" )
set(NBDNN_ICESWORD_BASE_SRC "")

# fetch noobsdnn header dir recursively
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD})
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD}/core/timer)
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD}/core/logger)
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD}/core/tensor)
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD}/core/operator)
noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD}/core/context)

if(USE_X86_PLACE)
    noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/operator/x86 "cpp" NBDNN_ICESWORD_BASE_SRC)
    noobsdnn_fetch_files_with_suffix(${NBDNN_ICESWORD}/operator/x86/kernel "cpp" NBDNN_ICESWORD_BASE_SRC)

    # compile cpp objs
    # add_library(NBDNN_ICESWORD_BASE_OBJS OBJECT ${NBDNN_ICESWORD_BASE_SRC})

    set(NBDNN_ICESWORD_TEMP_COMMMON_LIB ${noobsdnn_lib_so})

    # add icesword library to static
    if (BUILD_SHARED)
        add_library(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} SHARED ${NBDNN_ICESWORD_BASE_SRC})
        if(USE_MKLDNN OR USE_MKLML OR USE_XBYAK)
            message(STATUS ${NBDNN_ICESWORD_DEPENDENCIES})
            add_dependencies(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} ${NBDNN_ICESWORD_DEPENDENCIES})
        endif()
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES VERSION ${VERSION})
        target_link_libraries(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} ${NBDNN_LINKER_LIBS})
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${LIBRARY_OUTPUT_DIRECTORY})
    else()
        ADD_LIBRARY(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} STATIC ${NBDNN_ICESWORD_BASE_SRC})
        set_target_properties(${NBDNN_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${LIBRARY_OUTPUT_DIRECTORY})
    endif ()
endif()

