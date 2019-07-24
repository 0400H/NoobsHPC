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

file(REMOVE ${RUNTIME_OUTPUT_DIRECTORY}/unit_test/*)

set(NBHPC_LINKER_LIBS
    ${MKLDNN_LIB}
    ${MKLML_LIB}
    ${MKLML_IOMP_LIB}
)

message(STATUS "NoobsHPC linker libs: ${NBHPC_LINKER_LIBS}")

noobshpc_fetch_include_recursively(${NBHPC_UNIT_TEST})

if(USE_X86_PLACE)
    noobshpc_fetch_files_with_suffix(${NBHPC_UNIT_TEST} "cpp" NBHPC_TEST_CASE_SRC)

    # build test cases
    foreach(SRC_NAME ${NBHPC_TEST_CASE_SRC})
        #unpack the dir "/"
        string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
        list(GET SEXY_LIST -1 TEST_CASE_NAME)
        #get the file name without suffix
        string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
        list(GET SEXY_LIST 0 TEST_CASE_NAME)
        add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
        if(BUILD_SHARED)
            target_link_libraries(${TEST_CASE_NAME} ${noobshpc_lib_so} ${NBHPC_LINKER_LIBS} -ldl -lm)
        else()
            target_link_libraries(${TEST_CASE_NAME} -Wl,--whole-archive
                                  ${noobshpc_lib_static} -Wl,--no-whole-archive)
        endif()
        set_target_properties(${TEST_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
                              ${RUNTIME_OUTPUT_DIRECTORY}/unit_test)
    endforeach()
endif()


