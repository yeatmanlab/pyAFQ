from setuptools import find_packages
import versioneer

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Get metadata from the AFQ/_meta.py file:
ver_file = os.path.join('AFQ', '_meta.py')
with open(ver_file) as f:
    exec(f.read())

REQUIRES = []
with open('requirements.txt') as f:
    l = f.readline()[:-1]
    while l:
        REQUIRES.append(l)
        l = f.readline()[:-1]

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            packages=find_packages(),
            install_requires=REQUIRES,
            scripts=SCRIPTS,
            version=versioneer.get_version(),
            cmdclass=versioneer.get_cmdclass())


if __name__ == '__main__':
    setup(**opts)
