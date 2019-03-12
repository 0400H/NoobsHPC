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

file(REMOVE ${RUNTIME_OUTPUT_DIRECTORY}/benchmark/*)

set(NBDNN_LINKER_LIBS
    ${MKLDNN_LIB}
    ${MKLML_LIB}
    ${MKLML_IOMP_LIB}
)

message(STATUS "NoobsDNN linker libs: ${NBDNN_LINKER_LIBS}")

noobsdnn_fetch_include_recursively(${NBDNN_BENCHMARK})
noobsdnn_fetch_include_recursively(${NBDNN_BENCHMARK}/x86)

if(USE_X86_PLACE)
    noobsdnn_fetch_files_with_suffix(${NBDNN_BENCHMARK}/x86 "cpp" NBDNN_BENCH_CASE_SRC)

    # build src
    foreach(SRC_NAME ${NBDNN_BENCH_CASE_SRC})
        # unpack the dir "/"
        string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
        list(GET SEXY_LIST -1 BENCH_CASE_NAME)
        # get the file name without suffix
        string(REPLACE "." ";" SEXY_LIST ${BENCH_CASE_NAME})
        list(GET SEXY_LIST 0 BENCH_CASE_NAME)
        add_executable(${BENCH_CASE_NAME} ${SRC_NAME})
        if(BUILD_SHARED)
            target_link_libraries(${BENCH_CASE_NAME} ${noobsdnn_lib_so} ${NBDNN_LINKER_LIBS} -ldl -lm)
        else()
            target_link_libraries(${BENCH_CASE_NAME} -Wl,--whole-archive 
                                  ${noobsdnn_lib_static} ${NBDNN_LINKER_LIBS} -Wl,--no-whole-archive)
        endif()
        set_target_properties(${BENCH_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
                              ${RUNTIME_OUTPUT_DIRECTORY}/benchmark)
    endforeach()
endif()
