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

With both installation methods, you can include extensions to the base installation in square brackets. Change your working directory into the top-level directory of this repo
and issue::

  pip install -e .[dev,fury,afqbrowser,plot]

On some platforms, you may need to add quotes around the ``.[]`` part::

  pip install -e .'[dev,fury,afqbrowser,plot]'

.. note::

  The project follows the standard GitHub fork and pull request workflow. So if
  you plan on contributing to pyAFQ it is recommended that you fork the
  repository and issue pull requests. See :ref:`contributing`

.. note::

  It is also recommended that you utilize python virtual environment and
  package mangagement tools (e.g., conda) and begin with a clean environment.

.. note::

  Some of the examples in the documentation require additional dependencies. To install these, you can run `pip
  install pyAFQ[plot]`, which will include visualization tools that are required in these examples. For examples
  involving the cloudknot distributed computing library, you will also need to set up an [AWS account]([Create Account - aws.amazon.com](https://aws.amazon.com/resources/create-account/)) and have [docker](https://www.docker.com/) installed.


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
