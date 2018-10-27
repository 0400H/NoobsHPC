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

# used for temporary

noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD})
noobsdnn_fetch_include_recursively(${NBDNN_UTILS})

if(USE_GTEST)
    noobsdnn_find_gtest()
endif()

if(USE_X86_PLACE)
noobsdnn_fetch_files_with_suffix(${NBDNN_UNIT_TEST}/icesword/x86 "cpp" NBDNN_TEST_CASE_SRC)
endif()

file(REMOVE ${PROJECT_SOURCE_DIR}/output/unit_test/*)

# build test cases
foreach(SRC_NAME ${NBDNN_TEST_CASE_SRC})
	#unpack the dir "/"
	string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
	list(GET SEXY_LIST -1 TEST_CASE_NAME)
	#get the file name without suffix
	string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
	list(GET SEXY_LIST 0 TEST_CASE_NAME)
	add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
	if(BUILD_SHARED)
		target_link_libraries(${TEST_CASE_NAME} ${noobsdnn_lib_so})
	else()
		target_link_libraries(${TEST_CASE_NAME} -Wl,--whole-archive ${noobsdnn_lib_static} -Wl,--no-whole-archive)
	endif()
#	if(USE_ARM_PLACE)
#		target_link_libraries(${TEST_CASE_NAME} ${log-lib})
#	endif()
	set_target_properties(${TEST_CASE_NAME} PROPERTIES
						RUNTIME_OUTPUT_DIRECTORY 
						${PROJECT_SOURCE_DIR}/output/unit_test)
endforeach()
