
Using pyAFQ
===========

Organizing your data
~~~~~~~~~~~~~~~~~~~~

The pyAFQ software assumes that your data is organized according to a format
that emulates the `BIDS <http://bids.neuroimaging.io/>`_ format. Anatomical data
and segmentation are optional. If a T1-weighted anatomical image and its
segmentation are not provided, the software will use the diffusion data to
estimate the parts of the image that comprise the white matter.

.. note::

    The structure within the `raw` part of the BIDS directory doesn't matter
    much to the `pyAFQ` software, as the software will not touch this data at
    all.

As part of its operation, the software will create another directory, under
`study/derivatives/afq`, which will contain all of the derivatives created by
the software.

.. note::

    As the BIDS format for derivatives matures, we intend to update the pyAFQ
    software to support a fully BIDS-compliant set of derivatives. Both as
    inputs, as well as outputs.


Example data
------------

To get some example data formatted for input into `pyAFQ`, you can run::

    import AFQ.data as afd
    afd.organize_stanford_data()

This should create a folder in your home directory with a properly-formatted
data set in a directory called `stanford_hardi`.


Running the pyAFQ pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

`pyAFQ` provides a programmatic application programming interface that can
be used to integrate pyAFQ functionality into Python programs or into
interactive analysis sessions (e.g., using
`Jupyter <https://jupyter.org>`_ notebooks). Examples of using the API are
provided in the :ref:`examples` documentation section.

In addition, pyAFQ provides a command-line interface. After installing the
software, and organizing the data, run::

    pyAFQ /path/to/config.toml

pointing the program to the location of a configuration file. This will run
whole-brain tractography, segment the tracts, and extract tract-profiles for
each tract, generating a CSV file under
`study/derivatives/afq/tract_profiles.csv` that contains the tract profiles for
all participants/tracts/statistics.

The pyAFQ configuration file
----------------------------

This file should be a `toml <https://github.com/toml-lang/toml>`_ file. At
minimum, the file should contain information about the location of the
`dmriprep` folder::

    [files]
    dmriprep_folder = '/path/to/study/derivatives/dmriprep'


But additional configuration options can be provided.
See an example configuration file below::

    [EXAMPLE FILE HERE]
    
pyAFQ will store a copy of the configuration file alongside the computed
results. Note that the `title` variable and `[metadata]` section are both for
users to enter any title/metadata they would like and pyAFQ will generally
ignore them.

Usage tracking with Google Analytics
------------------------------------

To be able to assess usage of the software, we are recording each use of the
CLI as an event in Google Analytics, using `popylar <https://popylar.github.io>`_

The only information that we are recording is the fact that the CLI was called.
In addition, through Google Analytics, we will have access to very general
information, such as the country and city in which the computer using the CLI
was located and the time that it was used. At this time, we do not record any
additional information, although in the future we may want to record statistics
on the computational environment in which the CLI was called, such as the
operating system.

Opting out of this usage tracking can be done by calling the CLI with the
`--notrack` flag::

    pyAFQ /path/to/config.toml --notrack
