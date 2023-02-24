.. _installation_guide:


How to install ``pyAFQ``
==========================

The ``pyAFQ`` software works (at least) on Python 3.9 and 3.8.

How to install the release version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The released version of the software is the one that is officially
supported, and if you are getting started with ``pyAFQ``, this is
probably where you should get started

To install it, in a shell or command line, issue the following::

  pip install pyAFQ


How to install the development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The development version is probably less stable, but might include new
features and fixes. There are two ways to install this version. The first
uses ``pip``::

  pip install git+https://github.com/yeatmanlab/pyAFQ.git

The other requires that you clone the source code to your machine::

  git clone https://github.com/yeatmanlab/pyAFQ.git

Then, change your working directory into the top-level directory of this repo
and issue::

  pip install -e .[dev,fury,afqbrowser,plot]

which will install pyAFQ locally in editible mode and include extensions.

.. note::

  The project follows the standard GitHub fork and pull request workflow. So if
  you plan on contributing to pyAFQ it is recommended that you fork the
  repository and issue pull requests. See :ref:`contributing`

.. note::

  It is also recommended that you utilize python virtual environment and
  package mangagement tools and begin with a clean environment.


How to install using Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyAFQ automatically builds and pushes a Docker image with pyAFQ installed for every commit.
The images can be found `here <https://github.com/orgs/nrdg/packages/container/package/pyafq>`_
To pull an image, you can either pull the latest::

  docker pull ghcr.io/nrdg/pyafq:latest

or specify the commit using its hash::

  docker pull ghcr.io/nrdg/pyafq:41c03ce18fa2fd872ece9df72165e7d8d8f58baf

pyAFQ also automatically builds and pushes a Docker image with pyAFQ and
`QSIprep <https://qsiprep.readthedocs.io/en/latest/>`_ installed for every commit to master.
This image may be useful if you want an all-in-one image for pre-processing and tractometry.
You can pull the latest of this image or use a specific commit or tag as well::

  docker pull ghcr.io/nrdg/afqsi:latest
