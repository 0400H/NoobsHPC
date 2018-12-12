# add a target to generate API documentation with Doxygen

# set(DOXYGEN_EXECUTABLE doxygen)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
                      ${DOXYGEN_EXECUTABLE} ${NBDNN_CMAKE}/config/Doxyfile
                      WORKING_DIRECTORY ${NBDNN_OUTPUT}
                      COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
