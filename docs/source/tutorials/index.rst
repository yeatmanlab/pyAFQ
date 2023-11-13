.. _getting_started

Getting started with pyAFQ
--------------------------

First, install the software following :doc:`these instructions </howto/installation_guide>`.

Next, organize your preprocessed diffusion data according to the `BIDS <https://bids.neuroimaging.io>`_ standard.

Then, you are ready to run pyAFQ in one of the following ways:

1.  The first is as a program run in the command line. After installing the software, and organizing the data, run::

        pyAFQ /path/to/config.toml

    pointing the program to the location of a configuration file (see
    :doc:`configuration file specification </reference/config>` for an
    explanation of this file). This will run whole-brain tractography, segment
    the tracts, and extract tract-profiles for each tract, generating a CSV
    file under that contains the tract profiles for all
    participants/tracts/statistics.

2. The second is to write a program that uses `pyAFQ` as a software library.
   Detailed tutorials for this are provided in the next documentation section:


.. toctree::
    :maxdepth: 2

    tutorial_examples/index.rst