.. _installation_guide:


Installing ``pyAFQ``
==========================

The ``pyAFQ`` software works on Python 3.6 and 3.7.

Installing the release version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The released version of the software is the one that is officially
supported, and if you are getting started with ``pyAFQ``, this is
probably where you should get started

To install it, in a shell or command line, issue the following::

  pip install AFQ


Installing the development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The development version is probably less stable, but might include new
features and fixes. There are two ways to install this version. The first
uses ``pip``::

  pip install git+https://github.com/yeatmanlab/pyAFQ.git

The other requires that you clone the source code to your machine::

  git clone https://github.com/yeatmanlab/pyAFQ.git

Then, change your working directory into the top-level directory of this repo and issue::

  pip install .
