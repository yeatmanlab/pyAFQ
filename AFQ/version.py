import os.path as op
import glob

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pyAFQ: Automated Fiber Quantification ... in Python"
# Long description will go up on the pypi page
long_description = """

pyAFQ
========
Automated Fiber Quantification ... in Python.


License
=======
``pyAFQ`` is licensed under the terms of the BSD license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2015--, Jason Yeatman, Ariel Rokem, The University of Washington
eScience Institute.
"""

NAME = "pyAFQ"
MAINTAINER = "Jason Yeatman"
MAINTAINER_EMAIL = "jyeatman@uw.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/yeatmanlab/pyAFQ"
DOWNLOAD_URL = ""
LICENSE = "BSD"
AUTHOR = "Jason Yeatman "
AUTHOR_EMAIL = "jyeatman@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['AFQ',
            'AFQ.tests']
PACKAGE_DATA = {'AFQ': [op.join('data', '*')]}
REQUIRES = ["numpy", "scipy", "dipy"]
SCRIPTS = [op.join('bin', op.split(f)[-1]) for f in glob.glob('../bin/*')]
