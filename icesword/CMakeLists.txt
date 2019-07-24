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

set(NBHPC_ICESWORD_STATIC_RELAY "" )
set(NBHPC_ICESWORD_BASE_SRC "")

# fetch noobshpc header dir recursively
noobshpc_fetch_include_recursively(${NBHPC_ICESWORD})
noobshpc_fetch_include_recursively(${NBHPC_ICESWORD}/core/timer)
noobshpc_fetch_include_recursively(${NBHPC_ICESWORD}/core/logger)
noobshpc_fetch_include_recursively(${NBHPC_ICESWORD}/core/tensor)
noobshpc_fetch_include_recursively(${NBHPC_ICESWORD}/core/operator)
noobshpc_fetch_include_recursively(${NBHPC_ICESWORD}/core/context)

if(USE_X86_PLACE)
    noobshpc_fetch_files_with_suffix(${NBHPC_ICESWORD}/operator/x86 "cpp" NBHPC_ICESWORD_BASE_SRC)
    noobshpc_fetch_files_with_suffix(${NBHPC_ICESWORD}/operator/x86/kernel "cpp" NBHPC_ICESWORD_BASE_SRC)

    # compile cpp objs
    # add_library(NBHPC_ICESWORD_BASE_OBJS OBJECT ${NBHPC_ICESWORD_BASE_SRC})

    set(NBHPC_ICESWORD_TEMP_COMMMON_LIB ${noobshpc_lib_so})

    # add icesword library to static
    if (BUILD_SHARED)
        add_library(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} SHARED ${NBHPC_ICESWORD_BASE_SRC})
        if(USE_MKLDNN OR USE_MKLML OR USE_XBYAK)
            message(STATUS ${NBHPC_ICESWORD_DEPENDENCIES})
            add_dependencies(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} ${NBHPC_ICESWORD_DEPENDENCIES})
        endif()
        set_target_properties(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES VERSION ${VERSION})
        target_link_libraries(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} ${NBHPC_LINKER_LIBS})
        set_target_properties(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${LIBRARY_OUTPUT_DIRECTORY})
    else()
        ADD_LIBRARY(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} STATIC ${NBHPC_ICESWORD_BASE_SRC})
        set_target_properties(${NBHPC_ICESWORD_TEMP_COMMMON_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${LIBRARY_OUTPUT_DIRECTORY})
    endif ()
endif()

