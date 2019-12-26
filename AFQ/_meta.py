import os.path as op
import glob

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
SCRIPTS = [op.join('bin', op.split(f)[-1]) for f in glob.glob('bin/*')]
