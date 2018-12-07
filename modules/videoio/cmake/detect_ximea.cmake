if(NOT HAVE_XIMEA)
  if(WIN32)
    get_filename_component(regpath "[HKEY_CURRENT_USER\\Software\\XIMEA\\CamSupport\\API;Path]" ABSOLUTE)
  endif()
  if(X86_64)
    set(lib_dir "x64")
    set(lib_suffix "64")
  else()
    set(lib_dir "x86")
    set(lib_suffix "32")
  endif()
  find_path(XIMEA_INCLUDE "xiApi.h"
    PATHS "${XIMEA_ROOT}" ENV XIMEA_ROOT "/opt/XIMEA"
    HINTS "${regpath}"
    PATH_SUFFIXES "include" "API")
  find_library(XIMEA_LIBRARY m3api xiapi${lib_suffix}
    PATHS "${XIMEA_ROOT}" ENV XIMEA_ROOT "/opt/XIMEA"
    HINTS "${regpath}"
    PATH_SUFFIXES "API/${lib_dir}")
  if(XIMEA_INCLUDE AND XIMEA_LIBRARY)
    set(HAVE_XIMEA TRUE)
  endif()
endif()

if(HAVE_XIMEA)
  add_library(ocv::3rdparty::ximea INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::ximea PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${XIMEA_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "${XIMEA_LIBRARY}"
    INTERFACE_COMPILE_DEFINITIONS "HAVE_XIMEA")
endif()

set(HAVE_XIMEA ${HAVE_XIMEA} PARENT_SCOPE)
