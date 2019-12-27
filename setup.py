from setuptools import find_packages

import os.path as op

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = op.abspath(op.dirname(__file__))

# Get metadata from the AFQ/version.py file:
ver_file = op.join(here, 'AFQ', 'version.py')
with open(ver_file) as f:
    exec(f.read())

REQUIRES = []
with open(op.join(here, 'requirements.txt')) as f:
    ll = f.readline()[:-1]
    while ll:
        REQUIRES.append(ll)
        ll = f.readline()[:-1]

with open(op.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

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
            version=VERSION,
            python_requires=PYTHON_REQUIRES)


if __name__ == '__main__':
    setup(**opts)
