Using pyAFQ
===========

Organizing your data
~~~~~~~~~~~~~~~~~~~~

The pyAFQ software assumes that your data is organized according to a format
that emulates the `BIDS <http://bids.neuroimaging.io/>`_ format.

Preprocessed diffusion MRI data, as well as anatomical and segmentation data
should be organized as follows:

|    study
|      ├-derivatives
|            ├-dmriprep
|                ├── sub01
|                │   ├── sess01
|                │   │   ├── anat
|                │   │   │   ├── sub-01_sess-01_aparc+aseg.nii.gz
|                │   │   │   └── sub-01_sess-01_T1.nii.gz
|                │   │   └── dwi
|                │   │       ├── sub-01_sess-01_dwi.bvals
|                │   │       ├── sub-01_sess-01_dwi.bvecs
|                │   │       └── sub-01_sess-01_dwi.nii.gz
|                │   └── sess02
|                │       ├── anat
|                │       │   ├── sub-01_sess-02_aparc+aseg.nii.gz
|                │       │   └── sub-01_sess-02_T1w.nii.gz
|                │       └── dwi
|                │           ├── sub-01_sess-02_dwi.bvals
|                │           ├── sub-01_sess-02_dwi.bvecs
|                │           └── sub-01_sess-02_dwi.nii.gz
|                └── sub02
|                   ├── sess01
|                   │   ├── anat
|                   │       ├── sub-02_sess-01_aparc+aseg.nii.gz
|                   │   │   └── sub-02_sess-01_T1w.nii.gz
|                   │   └── dwi
|                   │       ├── sub-02_sess-01_dwi.bvals
|                   │       ├── sub-02_sess-01_dwi.bvecs
|                   │       └── sub-02_sess-01_dwi.nii.gz
|                   └── sess02
|                       ├── anat
|                       │   ├── sub-02_sess-02_aparc+aseg.nii.gz
|                       │   └── sub-02_sess-02_T1w.nii.gz
|                       └── dwi
|                           ├── sub-02_sess-02_dwi.bvals
|                           ├── sub-02_sess-02_dwi.bvecs
|                           └── sub-02_sess-02_dwi.nii.gz


Where the `raw` directory contains your raw data, and the `dmriprep` directory
contains data that has been processed to be motion-corrected, eddy-current
corrected and so on. Anatomical data is optional. If a T1-weighted anatomical
image and its segmentation are not provided (i.e., there is no `anat` folder),
the software will use the diffusion data to estimate the parts of the image that
comprise the white matter.

.. note::

    The structure within the `raw` directory doesn't matter much to the `pyAFQ`
    software, as the software will not touch this data at all.

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


But additional configuration options can be provided for the following values::

    title = "My AFQ analysis"

    [files]
    dmriprep_path = '/path/to/dmriprep/folder'

    [bundles]
    bundles = ['ATR', 'CGC', 'CST', 'HCC', 'IFO', 'ILF', 'SLF', 'ARC', 'UNC', 'FA', 'FP']
    seg_algo = 'AFQ'
    scalars_model = 'DTI'
    scalars = ['dti_fa', 'dti_md']

    [tracking]
    directions = 'det'
    max_angle = 30.0
    sphere = None
    seed_mask = None
    n_seeds = 1
    random_seeds = false
    stop_mask = None
    stop_threshold = 0
    step_size = 0.5
    min_length = 10
    max_length = 1000
    odf_model = 'DTI'
    wm_labels = [250, 251, 252, 253, 254, 255, 41, 2, 16, 77]

    [segmentation]
    nb_points = false
    seg_algo = 'AFQ'
    progressive = true
    greater_than = 50
    rm_small_clusters = 50
    model_clust_thr = 40
    reduction_thr = 40
    refine = false
    pruning_thr = 6
    b0_threshold = 0
    prob_threshold = 0
    rng = None
    return_idx = false
    filter_by_endpoints = true
    dist_to_aal = 4

    [cleaning]
    n_points = 100
    clean_rounds = 5
    distance_threshold = 5
    length_threshold = 4
    min_sl = 20
    stat = 'mean'
    return_idx = false

    [compute]
    dask_it = false

    [metadata]
    a_string = "A string with some description"
    list_of_values = ["val1", 1, 2, 3]
    some_boolean = true

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