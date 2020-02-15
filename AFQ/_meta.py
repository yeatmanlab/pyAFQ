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

NAME = "pyAFQ"
MAINTAINER = "Ariel Rokem"
MAINTAINER_EMAIL = "arokem@gmail.com"
DESCRIPTION = description
URL = "http://github.com/yeatmanlab/pyAFQ"
DOWNLOAD_URL = ""
LICENSE = "BSD"
AUTHOR = "pyAFQ developers"
AUTHOR_EMAIL = "arokem@gmail.com"
PLATFORMS = "OS Independent"
SCRIPTS = [op.join('bin', op.split(f)[-1]) for f in glob.glob('bin/*')]
PYTHON_REQUIRES='>=3.6'