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
  add_3p_target(cap_ios "" "${libs}" "HAVE_CAP_IOS")
endif()

set(HAVE_CAP_IOS ${HAVE_CAP_IOS} PARENT_SCOPE)
