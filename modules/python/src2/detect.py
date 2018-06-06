#!/usr/bin/env python

from __future__ import print_function
import os
os.environ['DISTUTILS_USE_SDK']='1'

import numpy.distutils.misc_util as npdu
import distutils.sysconfig as du
from pprint import pprint
import site
import sys

print("EXECUTABLE = ", sys.executable)
print("VERSION_STRING = {}.{}".format(sys.version_info.major, sys.version_info.minor))
print("VERSION_MAJOR =", sys.version_info.major)
print("VERSION_MINOR =", sys.version_info.minor)
print("LIBRARIES =", du.get_config_var('LDLIBRARY'))
print("INCLUDE_PATH =", du.get_config_var('INCLUDEPY'))
print("PACKAGES_PATH_USER =", site.getusersitepackages())
print("PACKAGES_PATH =", site.getsitepackages())
print("CVPY_SUFFIX =", du.get_config_var('SO'))
print("NUMPY_INCLUDE_DIRS =", npdu.get_numpy_include_dirs())

with open('dump.txt', 'w') as out:
    v = du.get_config_vars()
    pprint(v, out)
