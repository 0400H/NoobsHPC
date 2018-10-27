noobsdnn_fetch_include_recursively(${NBDNN_ICESWORD})
noobsdnn_fetch_include_recursively(${NBDNN_UTILS})

message(STATUS "Benchmark source files: ${DIR_BENCHMARK_SRCS}")

set(NBDNN_LINKER_LIBS
    ${MKLDNN_LIB}
    ${MKLML_LIB}
    ${MKLML_IOMP_LIB}
)
message(STATUS "NoobsDNN linker libs: ${NBDNN_LINKER_LIBS}")

if(USE_X86_PLACE)
	aux_source_directory(x86 DIR_X86_SRCS)
	list(REMOVE_ITEM DIR_X86_SRCS x86/benchmark_common.cpp)
	set(BENCHMARK_COMMONS x86/benchmark_common.cpp)
	message(STATUS "benchmark x86 source files: ${DIR_X86_SRCS}")
	# build src
	foreach(SRC_NAME ${DIR_X86_SRCS})
		#unpack the dir "/"
		string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
		list(GET SEXY_LIST -1 BENCHMARK_NAME)
		#get the file name without suffix
		string(REPLACE "." ";" SEXY_LIST ${BENCHMARK_NAME})
		list(GET SEXY_LIST 0 BENCHMARK_NAME)
		add_executable(${BENCHMARK_NAME}  ${SRC_NAME} ${BENCHMARK_COMMONS})
		if(BUILD_SHARED)
			target_link_libraries(${BENCHMARK_NAME} ${noobsdnn_lib_so} ${NBDNN_LINKER_LIBS} -ldl -lm)
		else()
			target_link_libraries(${BENCHMARK_NAME} icesword ${NBDNN_LINKER_LIBS} -ldl -lm)
		endif()
		set_target_properties(${BENCHMARK_NAME} PROPERTIES
                                      RUNTIME_OUTPUT_DIRECTORY 
                                      ${PROJECT_SOURCE_DIR}/output/benchmark)
	endforeach()
endif()
