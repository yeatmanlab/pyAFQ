
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

    title = "My AFQ analysis"

    [BIDS]
    
    # The path to preprocessed diffusion data organized in a BIDS
    # dataset. This should contain a BIDS derivative dataset with
    # preprocessed dwi/bvals/bvecs.
    bids_path = ''
    
    # The name of the pipeline used to preprocess the DWI data.
    # Default: "dmriprep".
    dmriprep = 'dmriprep'
    
    # The name of the pipeline used to generate
    # a segmentation image.
    # Default: "dmriprep"
    segmentation = 'dmriprep'
    
    # The suffix that identifies the segmentation image.
    # Default: "seg".
    seg_suffix = 'seg'
    
    [REGISTRATION]
    
    # The value of b under which
    # it is considered to be b0. Default: 0.
    b0_threshold = 0
    
    # Minimum b value you want to use
    # from the dataset (other than b0).
    # If None, there is no minimum limit. Default: None
    min_bval = ''
    
    # Maximum b value you want to use
    # from the dataset (other than b0).
    # If None, there is no maximum limit. Default: None
    max_bval = ''
    
    # The source image data to be registered.
    # Can either be a Nifti1Image, a path to a Nifti1Image, or
    # If "b0", "dti_fa_subject", "subject_sls", or "power_map,"
    # image data will be loaded automatically.
    # If "subject_sls" is used, slr registration will be used
    # and reg_template should be "hcp_atlas".
    # Default: "b0"
    reg_subject = 'b0'
    
    # The target image data for registration.
    # Can either be a Nifti1Image, a path to a Nifti1Image, or
    # If "mni_T2", "dti_fa_template", "hcp_atlas", or "mni_T1",
    # image data will be loaded automatically.
    # If "hcp_atlas" is used, slr registration will be used
    # and reg_subject should be "subject_sls".
    # Default: "mni_T2"
    reg_template = 'mni_T2'
    
    # Whether to mask the chosen template(s)
    # with a brain-mask.
    # Default: True
    mask_template = true
    
    # This is either a list of the labels of
    # the white matter in the segmentation file or (if a float is
    # provided) the threshold FA to use for creating the white-matter
    # mask. For example, the white matter values for the segmentation
    # provided with the HCP data including labels for midbrain are:
    wm_criterion = 0.1
    
    # Whether to perform pre-alignment before perforiming
    # the diffeomorphic mapping in registration. Default: True
    use_prealign = true
    
    [COMPUTE]
    
    # Whether to use a dask DataFrame object
    dask_it = false
    
    # Whether to recompute or ignore existing derivatives.
    # This parameter can be turned on/off dynamically.
    # Default: False
    force_recompute = false
    
    [BUNDLES]
    
    # List of scalars to use.
    # Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md"
    # Default:
    scalars = ['dti_fa', 'dti_md']
    
    [VIZ]
    
    # Whether to use a virtual fram buffer. This is neccessary if
    # generating GIFs in a headless environment. Default: False
    virtual_frame_buffer = false
    
    # Which visualization backend to us.
    # One of {"fury", "plotly"}.
    # Default: "fury"
    viz_backend = 'fury'
    
    [TRACTOGRAPHY]
    
    # Full path to a nifti file containing CSD spherical harmonic
    # coefficients, or nibabel img with model params.
    params_file = ''
    
    # How tracking directions are determined.
    # One of: {"deterministic" | "probablistic"}
    directions = 'det'
    
    # The maximum turning angle in each step. Default: 30
    max_angle = 30.0
    
    # The discretization of direction getting. default:
    # dipy.data.default_sphere.
    sphere = ''
    
    # Float or binary mask describing the ROI within which we seed for
    # tracking.
    # Default to the entire volume (all ones).
    seed_mask = ''
    
    # A value of the stop_mask below which tracking is terminated.
    # Default to 0.
    seed_threshold = 0
    
    # The seeding density: if this is an int, it is is how many seeds in each
    # voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
    # array, these are the coordinates of the seeds. Unless random_seeds is
    # set to True, in which case this is the total number of random seeds
    # to generate within the mask.
    n_seeds = 1
    
    # Whether to generate a total of n_seeds random seeds in the mask.
    # Default: XXX.
    random_seeds = false
    
    # random seed used to generate random seeds if random_seeds is
    # set to True. Default: None
    rng_seed = ''
    
    # A float or binary mask that determines a stopping criterion (e.g. FA).
    # Default to no stopping (all ones).
    stop_mask = ''
    
    # A value of the stop_mask below which tracking is terminated. Default to
    # 0 (this means that if no stop_mask is passed, we will stop only at
    # the edge of the image)
    stop_threshold = 0
    
    # The size (in mm) of a step of tractography. Default: 1.0
    step_size = 0.5
    
    # The miminal length (mm) in a streamline. Default: 10
    min_length = 10
    
    # The miminal length (mm) in a streamline. Default: 250
    max_length = 1000
    
    # One of {"DTI", "CSD"}. Defaults to use "DTI"
    odf_model = 'DTI'
    
    [SEGMENTATION]
    
    # Resample streamlines to nb_points number of points.
    # If False, no resampling is done. Default: False
    nb_points = false
    
    # Algorithm for segmentation (case-insensitive):
    # 'AFQ': Segment streamlines into bundles,
    # based on inclusion/exclusion ROIs.
    # 'Reco': Segment streamlines using the RecoBundles algorithm
    # [Garyfallidis2017].
    # Default: 'AFQ'
    seg_algo = 'AFQ'
    
    # Using RecoBundles Algorithm.
    # Whether or not to use progressive technique
    # during whole brain SLR.
    # Default: True.
    progressive = true
    
    # Using RecoBundles Algorithm.
    # Keep streamlines that have length greater than this value
    # during whole brain SLR.
    # Default: 50.
    greater_than = 50
    
    # Using RecoBundles Algorithm.
    # Remove clusters that have less than this value
    # during whole brain SLR.
    # Default: 50
    rm_small_clusters = 50
    
    # Using AFQ Algorithm.
    # All b-values with values less than or equal to `bo_threshold` are
    # considered as b0s i.e. without diffusion weighting.
    # Default: 0.
    b0_threshold = 0
    
    # Using AFQ Algorithm.
    # Initial cleaning of fiber groups is done using probability maps
    # from [Hua2008]_. Here, we choose an average probability that
    # needs to be exceeded for an individual streamline to be retained.
    # Default: 0.
    prob_threshold = 0
    
    # If None, creates RandomState. Used in RecoBundles Algorithm.
    # Default: None.
    rng = ''
    
    # Whether to return the indices in the original streamlines as part
    # of the output of segmentation.
    return_idx = false
    
    # Whether to filter the bundles based on their endpoints relative
    # to regions defined in the AAL atlas. Applies only to the waypoint
    # approach (XXX for now). Default: True.
    filter_by_endpoints = true
    
    # If filter_by_endpoints is True, this is the distance from the
    # endpoints to the AAL atlas ROIs that is required.
    dist_to_aal = 4
    
    # The full path to a folder into which intermediate products
    # are saved. Default: None, means no saving of intermediates.
    save_intermediates = ''
    
    [CLEANING]
    
    # A whole-brain tractogram to be segmented.
    tg = ''
    
    # Number of rounds of cleaning based on the Mahalanobis distance from
    # the mean of extracted bundles. Default: 5
    clean_rounds = 5
    
    # Threshold of cleaning based on the Mahalanobis distance (the units are
    # standard deviations). Default: 5.
    distance_threshold = 5
    
    # Threshold for cleaning based on length (in standard deviations). Length
    # of any streamline should not be *more* than this number of stdevs from
    # the mean length.
    length_threshold = 4
    
    # Number of streamlines in a bundle under which we will
    # not bother with cleaning outliers. Default: 20.
    min_sl = 20
    
    # The statistic of each node relative to which the Mahalanobis is
    # calculated. Default: `np.mean` (but can also use median, etc.)
    stat = 'mean'
    
    # Whether to return indices in the original streamlines.
    # Default: False.
    return_idx = false
    
    
    
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
