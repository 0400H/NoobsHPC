# Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
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
noobsdl_fetch_include_recursively(${NBDL_SABER})
noobsdl_fetch_include_recursively(${NBDL_UTILS})
noobsdl_fetch_include_recursively(${NBDL_FRAMEWORK}/core)


set(NBDL_BASE_SRC "")

# add ak_base_source files
noobsdl_fetch_files_with_suffix(${NBDL_FRAMEWORK}/core "cpp" NBDL_BASE_SRC)
noobsdl_fetch_files_with_suffix(${NBDL_FRAMEWORK}/core/net "cpp" NBDL_BASE_SRC)
noobsdl_fetch_files_with_suffix(${NBDL_FRAMEWORK}/core/operator "cpp" NBDL_BASE_SRC)


list(APPEND NBDL_SRC ${NBDL_BASE_SRC})
unset(NBDL_BASE_SRC)

# add library to shared or static
if(UNIX OR APPLE)
	if(BUILD_SHARED)
		add_library(${noobsdl_lib_so} SHARED ${NBDL_SRC})
		add_dependencies(${noobsdl_lib_so} ${NBDL_SABER_LIB_TARGET})
		# set shared lib version
		set_target_properties(${noobsdl_lib_so} PROPERTIES VERSION ${VERSION})
        
		target_link_libraries(${noobsdl_lib_so} ${NBDL_SABER_LIB_TARGET} ${NBDL_LINKER_LIBS})
		set_target_properties(${noobsdl_lib_so} PROPERTIES LINK_FLAGS "")
		set_target_properties(${noobsdl_lib_so} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)
        install(DIRECTORY ${NBDL_FRAMEWORK} ${NBDL_SABER} ${NBDL_UTILS}
            	DESTINATION ${PROJECT_SOURCE_DIR}/output/
            	FILES_MATCHING 
				PATTERN "*.h"
				PATTERN "*.inl")
	endif()
	if(BUILD_STATIC)
		add_library(${noobsdl_lib_static} STATIC ${NBDL_SRC})
		add_dependencies(${noobsdl_lib_static} ${NBDL_SABER_LIB_TARGET})# ${noobsdl_framework_static})
		#set_target_properties(${noobsdl_lib_static} PROPERTIES VERSION ${VERSION})
		target_link_libraries(${noobsdl_lib_static} ${NBDL_SABER_LIB_TARGET} ${NBDL_LINKER_LIBS})
        set_target_properties(${noobsdl_lib_static} PROPERTIES LINK_FLAGS "")
		set_target_properties(${noobsdl_lib_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/)
       	install(DIRECTORY ${NBDL_FRAMEWORK} ${NBDL_SABER} ${NBDL_UTILS}
                DESTINATION ${PROJECT_SOURCE_DIR}/output/
                FILES_MATCHING 
				PATTERN "*.h"
				PATTERN "*.inl")
	endif()
endif()
