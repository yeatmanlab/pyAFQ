Using pyAFQ
===========

`pyAFQ` provides a programmatic application programming interface (API) that
can be used to integrate pyAFQ functionality into Python programs or into
interactive analysis sessions (e.g., using
`Jupyter <https://jupyter.org>`_ notebooks). Examples of using the API are
provided in the :ref:`examples` documentation section.

In addition, pyAFQ provides a command-line interface (CLI). After installing
the software, and organizing the data, run::

    pyAFQ /path/to/config.toml

pointing the program to the location of a configuration file. This will run
whole-brain tractography, segment the tracts, and extract tract-profiles for
each tract, generating a CSV file under
`study/derivatives/afq/tract_profiles.csv` that contains the tract profiles for
all participants/tracts/statistics.
The csv file has the same format as
`nodes.csv in AFQ Browser <https://yeatmanlab.github.io/AFQ-Browser/dataformat.html>`_.


.. toctree::
    :hidden:
    :maxdepth: 2

    data
    usage
    config
    viz_backend
    mask
    mapping
    scalars
    converter
    tracking
