if(APPLE AND IOS)
  set(HAVE_CAP_IOS TRUE)
  set(libs
    "-framework Accelerate"
    "-framework AVFoundation"
    "-framework CoreGraphics"
    "-framework CoreImage"
    "-framework CoreMedia"
    "-framework CoreVideo"
    "-framework QuartzCore"
    "-framework UIKit")
  add_library(ocv::3rdparty::cap_ios INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::cap_ios PROPERTIES
    INTERFACE_LINK_LIBRARIES "${libs}"
    INTERFACE_COMPILE_DEFINITIONS "HAVE_CAP_IOS")
endif()

set(HAVE_CAP_IOS ${HAVE_CAP_IOS} PARENT_SCOPE)
