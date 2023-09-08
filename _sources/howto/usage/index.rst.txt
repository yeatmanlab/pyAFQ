How to use `pyAFQ`
==================

`pyAFQ` can be used in two different ways.

1. The first is as a program run in the command line. After installing the software, and organizing the data, run::

    pyAFQ /path/to/config.toml

pointing the program to the location of a configuration file (see :ref:`config`
for an explanation of this file). This will run whole-brain tractography,
segment the tracts, and extract tract-profiles for each tract, generating a CSV
file under `study/derivatives/afq/tract_profiles.csv` that contains the tract
profiles for all participants/tracts/statistics. The csv file has the same
format as `nodes.csv in AFQ Browser <https://yeatmanlab.github.io/AFQ-Browser/dataformat.html>`_.


2. The second is to write a program that uses `pyAFQ` as a software library.
This is because `pyAFQ` provides a programmatic application programming
interface (API) that can be used to integrate pyAFQ functionality into Python
programs or into interactive analysis sessions (e.g., using `Jupyter <https://jupyter.org>`_ notebooks).
Examples of using the API are provided in the :ref:`examples` documentation section.

.. toctree::
    :maxdepth: 2

    data
    usage
    kwargs
    methods
    bundledict
    config
    viz_backend
    mask
    mapping
    scalars
    converter
    tracking
    docker
