if(APPLE)
  set(HAVE_AVFOUNDATION TRUE)
  if(IOS)
    set(libs "-framework AVFoundation" "-framework QuartzCore")
  else()
    set(libs
      "-framework Cocoa"
      "-framework Accelerate"
      "-framework AVFoundation"
      "-framework CoreGraphics"
      "-framework CoreMedia"
      "-framework CoreVideo"
      "-framework QuartzCore")
  endif()
  add_library(ocv::3rdparty::avfoundation INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::avfoundation PROPERTIES
    INTERFACE_LINK_LIBRARIES "${libs}"
    INTERFACE_COMPILE_DEFINITIONS "HAVE_AVFOUNDATION")
endif()

set(HAVE_AVFOUNDATION ${HAVE_AVFOUNDATION} PARENT_SCOPE)
