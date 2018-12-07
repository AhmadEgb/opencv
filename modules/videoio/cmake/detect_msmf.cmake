# --- VideoInput/Microsoft Media Foundation ---
if(NOT HAVE_MSMF)
  check_include_file(mfapi.h HAVE_MFAPI)
  if(HAVE_MFAPI)
    set(HAVE_MSMF TRUE)
  endif()
endif()

if(HAVE_MSMF)
  check_include_file(d3d11.h HAVE_D3D11)
  check_include_file(d3d11_4.h HAVE_D3D11_4)
  set(defs "HAVE_MSMF")
  if(HAVE_D3D11 AND HAVE_D3D11_4)
    list(APPEND defs "HAVE_DXVA")
  endif()
  add_library(ocv::3rdparty::msmf INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::msmf PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${defs}")
endif()

set(HAVE_MSMF ${HAVE_MSMF} PARENT_SCOPE)
