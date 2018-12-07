# --- VideoInput/DirectShow ---
if(NOT HAVE_DSHOW AND MSVC AND NOT MSVC_VERSION LESS 1500)
  set(HAVE_DSHOW TRUE)
endif()

if(NOT HAVE_DSHOW)
  check_include_file(dshow.h HAVE_DSHOW)
endif()

if(HAVE_DSHOW)
  add_library(ocv::3rdparty::dshow INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::dshow PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "HAVE_DSHOW")
endif()

set(HAVE_DSHOW ${HAVE_DSHOW} PARENT_SCOPE)
