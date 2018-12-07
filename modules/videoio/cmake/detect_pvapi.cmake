# --- PvApi ---
if(NOT HAVE_PVAPI)
  if(X86_64)
    set(arch x64)
  else()
    set(arch x86)
  endif()
  find_path(PVAPI_INCLUDE "PvApi.h"
    PATHS "${PVAPI_ROOT}" ENV PVAPI_ROOT
    PATH_SUFFIXES "inc-pc")
  find_library(PVAPI_LIBRARY "PvAPI"
    PATHS "${PVAPI_ROOT}" ENV PVAPI_ROOT
    PATH_SUFFIXES "bin-pc/${arch}/${gcc}")
  if(PVAPI_INCLUDE AND PVAPI_LIBRARY)
    set(HAVE_PVAPI TRUE)
  endif()
endif()

if(HAVE_PVAPI)
  add_library(ocv::3rdparty::pvapi INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::pvapi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PVAPI_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "${PVAPI_LIBRARY}"
  INTERFACE_COMPILE_DEFINITIONS "HAVE_PVAPI")
endif()

set(HAVE_PVAPI ${HAVE_PVAPI} PARENT_SCOPE)
