
aux_source_directory(. DIR_BASE_SRCS)

if(USE_X86)
    aux_source_directory(impl/x86 X86_BASE_SRCS)
    aux_source_directory(impl/x86/kernel X86_BASE_SRCS_KERNEL)
    aux_source_directory(impl/x86/math X86_BASE_SRCS_MATH)
endif()

set(DIR_SRCS_CUR "")

foreach(SRC_NAME ${DIR_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${X86_BASE_SRCS})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${X86_BASE_SRCS_KERNEL})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

foreach(SRC_NAME ${X86_BASE_SRCS_MATH})
    #unpack the dir "/"
    string(REPLACE "./" "" FILE_NAME ${SRC_NAME})
    string(REPLACE " " "" FILE_NAME ${FILE_NAME})
    #string(REPLACE ".cpp" ".cpp;" FILE_NAME ${FILE_NAME})
    list(APPEND DIR_SRCS_CUR "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}")
endforeach()

message(STATUS "base source files:${DIR_SRCS_CUR}")

set(DIRSRCS "${DIRSRCS}${DIR_SRCS_CUR};" PARENT_SCOPE)

