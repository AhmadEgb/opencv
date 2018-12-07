if(NOT HAVE_MFX)
  set(paths "${MFX_HOME}" ENV "MFX_HOME" ENV "INTELMEDIASDKROOT")
  if(MSVC)
    if(MSVC_VERSION LESS 1900)
      set(vs_suffix)
    else()
      set(vs_suffix "_vs2015")
    endif()
    if(X86_64)
      set(vs_arch "x64")
    else()
      set(vs_arch "win32")
    endif()
  endif()
  find_path(MFX_INCLUDE mfxdefs.h
    PATHS ${paths}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH)
  find_library(MFX_LIBRARY mfx libmfx${vs_suffix}
    PATHS ${paths}
    PATH_SUFFIXES "lib64" "lib/lin_x64" "lib/${vs_arch}"
    NO_DEFAULT_PATH)
  if(MFX_INCLUDE AND MFX_LIBRARY)
    set(HAVE_MFX TRUE)
    set(MFX_INCLUDE_DIRS "${MFX_INCLUDE}")
    set(MFX_LIBRARIES "${MFX_LIBRARY}")
  endif()
endif()

if(HAVE_MFX AND UNIX)
  find_path(MFX_va_INCLUDE va/va.h PATHS ${paths} PATH_SUFFIXES "include")
  find_library(MFX_va_LIBRARY va PATHS ${paths} PATH_SUFFIXES "lib64" "lib/lin_x64")
  find_library(MFX_va_drm_LIBRARY va-drm PATHS ${paths} PATH_SUFFIXES "lib64" "lib/lin_x64")
  if(MFX_va_INCLUDE AND MFX_va_LIBRARY AND MFX_va_drm_LIBRARY)
    list(APPEND MFX_INCLUDE_DIRS "${MFX_va_INCLUDE}")
    list(APPEND MFX_LIBRARIES "${MFX_va_LIBRARY}" "${MFX_va_drm_LIBRARY}")
    # list(APPEND MFX_LIBRARIES "-Wl,--exclude-libs=libmfx")
  else()
    set(HAVE_MFX FALSE)
  endif()
endif()

if(HAVE_MFX)
  add_library(ocv::3rdparty::mediasdk INTERFACE IMPORTED)
  set_target_properties(ocv::3rdparty::mediasdk PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MFX_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${MFX_LIBRARIES}"
    INTERFACE_COMPILE_DEFINITIONS "HAVE_MFX")
endif()

set(HAVE_MFX ${HAVE_MFX} PARENT_SCOPE)
