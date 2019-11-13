# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Base module variables."""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__packagename__ = 'pyAFQ'
__copyright__ = 'Copyright 2019, The pyAFQ developers'
__credits__ = ('Contributors: please check the ``.zenodo.json`` file at the top-level folder'
               'of the repository')
__url__ = 'https://github.com/yeatmanlab/pyAFQ'

DOWNLOAD_URL = (
    'https://github.com/yeatmanlab/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))
