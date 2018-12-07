# --- V4L ---
if(NOT HAVE_V4L)
  set(CMAKE_REQUIRED_QUIET TRUE) # for check_include_file
  check_include_file(linux/videodev2.h HAVE_CAMV4L2)
  check_include_file(sys/videoio.h HAVE_VIDEOIO)
  if(HAVE_CAMV4L2 OR HAVE_VIDEOIO)
    set(HAVE_V4L TRUE)
    add_library(ocv::3rdparty::v4l INTERFACE IMPORTED)
    if(HAVE_CAMV4L2)
      set_target_properties(ocv::3rdparty::v4l PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "HAVE_CAMV4L2")
    endif()
    if(HAVE_VIDEOIO)
      set_target_properties(ocv::3rdparty::v4l PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "HAVE_VIDEOIO")
    endif()
  endif()
endif()

set(HAVE_V4L ${HAVE_V4L} PARENT_SCOPE)
