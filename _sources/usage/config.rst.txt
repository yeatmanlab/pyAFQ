
The pyAFQ configuration file
----------------------------

This file should be a `toml <https://github.com/toml-lang/toml>`_ file. At
minimum, the file should contain the BIDS path::

    [files]
    bids_path = "path/to/study"


But additional configuration options can be provided.
See an example configuration file below::

    title = "My AFQ analysis"

    # Initialize a GroupAFQ object from a BIDS dataset.
    
    # Use '' to indicate None
    # Wrap dictionaries in quotes
    # Wrap definition object instantiations in quotes
    
    [BIDS_PARAMS]
    
    # The path to preprocessed diffusion data organized in a BIDS
    # dataset. This should contain a BIDS derivative dataset with
    # preprocessed dwi/bvals/bvecs.
    bids_path = ""
    
    # Filter to pass to bids_layout.get when finding DWI files.
    # Default: {"suffix": "dwi"}
    bids_filters = "{'suffix': 'dwi'}"
    
    # The name of the pipeline used to preprocess the DWI data.
    # Default: "all".
    preproc_pipeline = "all"
    
    # List of participant labels (subject IDs) to perform
    # processing on. If None, all subjects are used.
    # Default: None
    participant_labels = ""
    
    # Path to output directory. If None, outputs are put
    # in a AFQ pipeline folder in the derivatives folder of
    # the BIDS directory. pyAFQ will use existing derivatives
    # from the output directory if they exist, instead of recalculating
    # them (this means you need to clear the output folder if you want
    # to recalculate a derivative).
    # Default: None
    output_dir = ""
    
    # Parameters to pass to paramap in AFQ.utils.parallel,
    # to parallelize computations across subjects and sessions.
    # Set "n_jobs" to -1 to automatically parallelize as
    # the number of cpus. Here is an example for how to do
    # multiprocessing with 4 cpus:
    # {"n_jobs": 4, "engine": "joblib", "backend": "loky"}
    # Default: {"engine": "serial"}
    parallel_params = "{'engine': 'serial'}"
    
    # Additional arguments to give to BIDSLayout from pybids.
    # For large datasets, try:
    # {"validate": False, "index_metadata": False}
    # Default: {}
    bids_layout_kwargs = "{}"
    
    [TRACTOGRAPHY_PARAMS]
    
    # How tracking directions are determined.
    # One of: {"det" | "prob"}
    # Default: "prob"
    directions = "prob"
    
    # The maximum turning angle in each step. Default: 30
    max_angle = 30.0
    
    # The discretization of direction getting. default:
    # dipy.data.default_sphere.
    sphere = ""
    
    # Float or binary mask describing the ROI within which we seed for
    # tracking.
    # Default to the entire volume (all ones).
    seed_mask = ""
    
    # A value of the seed_mask below which tracking is terminated.
    # Default to 0.
    seed_threshold = 0
    
    # The seeding density: if this is an int, it is is how many seeds in each
    # voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
    # array, these are the coordinates of the seeds. Unless random_seeds is
    # set to True, in which case this is the total number of random seeds
    # to generate within the mask. Default: 1
    n_seeds = 1
    
    # Whether to generate a total of n_seeds random seeds in the mask.
    # Default: False.
    random_seeds = false
    
    # random seed used to generate random seeds if random_seeds is
    # set to True. Default: None
    rng_seed = ""
    
    # If array: A float or binary mask that determines a stopping criterion
    # (e.g. FA).
    # If tuple: it contains a sequence that is interpreted as:
    # (pve_wm, pve_gm, pve_csf), each item of which is either a string
    # (full path) or a nibabel img to be used in particle filtering
    # tractography.
    # A tuple is required if tracker is set to "pft".
    # Defaults to no stopping (all ones).
    stop_mask = ""
    
    # If float, this a value of the stop_mask below which tracking is
    # terminated (and stop_mask has to be an array).
    # If str, "CMC" for Continuous Map Criterion [Girard2014]_.
    # "ACT" for Anatomically-constrained tractography [Smith2012]_.
    # A string is required if the tracker is set to "pft".
    # Defaults to 0 (this means that if no stop_mask is passed,
    # we will stop only at the edge of the image).
    stop_threshold = 0
    
    # The size of a step (in mm) of tractography. Default: 0.5
    step_size = 0.5
    
    # The miminal length (mm) in a streamline. Default: 20
    min_length = 50
    
    # The miminal length (mm) in a streamline. Default: 250
    max_length = 250
    
    # One of {"DTI", "CSD", "DKI"}. Defaults to use "DTI"
    odf_model = "CSD"
    
    # Which strategy to use in tracking. This can be the standard local
    # tracking ("local") or Particle Filtering Tracking ([Girard2014]_).
    # One of {"local", "pft"}. Default: "local"
    tracker = "local"
    
    [SEGMENTATION_PARAMS]
    
    # Resample streamlines to nb_points number of points.
    # If False, no resampling is done. Default: False
    nb_points = false
    
    # Subsample streamlines to nb_streamlines.
    # If False, no subsampling is don. Default: False
    nb_streamlines = false
    
    # Algorithm for segmentation (case-insensitive):
    # 'AFQ': Segment streamlines into bundles,
    # based on inclusion/exclusion ROIs.
    # 'Reco': Segment streamlines using the RecoBundles algorithm
    # [Garyfallidis2017].
    # Default: 'AFQ'
    seg_algo = "AFQ"
    
    # Algorithm for streamline registration (case-insensitive):
    # 'slr' : Use Streamlinear Registration [Garyfallidis2015]_
    # 'syn' : Use image-based nonlinear registration
    # If None, will use SyN if a mapping is provided, slr otherwise.
    # If  seg_algo="AFQ", SyN is always used.
    # Default: None
    reg_algo = ""
    
    # Whether to clip the streamlines to be only in between the ROIs.
    # Default: False
    clip_edges = false
    
    # How to parallelize segmentation across processes when performing
    # waypoint ROI segmentation. Set to {"engine": "serial"} to not
    # perform parallelization. See ``AFQ.utils.parallel.pafor`` for
    # details.
    # Default: {"n_jobs": -1, "engine": "joblib",
    # "backend": "loky"}
    parallel_segmentation = "{'n_jobs': -1, 'engine': 'joblib', 'backend': 'loky'}"
    
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
    
    # Parameter passed on to recognize for Recobundles.
    # See Recobundles documentation.
    # Default: 1.25
    model_clust_thr = 1.25
    
    # Parameter passed on to recognize for Recobundles.
    # See Recobundles documentation.
    # Default: 25
    reduction_thr = 25
    
    # Parameter passed on to recognize for Recobundles.
    # See Recobundles documentation.
    # Default: False
    refine = false
    
    # Parameter passed on to recognize for Recobundles.
    # See Recobundles documentation.
    # Default: 12
    pruning_thr = 12
    
    # Using AFQ Algorithm.
    # All b-values with values less than or equal to `bo_threshold` are
    # considered as b0s i.e. without diffusion weighting.
    # Default: 50.
    b0_threshold = 50
    
    # Using AFQ Algorithm.
    # Initial cleaning of fiber groups is done using probability maps
    # from [Hua2008]_. Here, we choose an average probability that
    # needs to be exceeded for an individual streamline to be retained.
    # Default: 0.
    prob_threshold = 0
    
    # Whether to use distance from nearest ROI as a tie breaker when a
    # streamline qualifies as a part of multiple bundles. If False,
    # probability maps are used.
    # Default : False.
    roi_dist_tie_break = false
    
    # The distance that a streamline node has to be from the waypoint
    # ROI in order to be included or excluded.
    # If set to None (default), will be calculated as the
    # center-to-corner distance of the voxel in the diffusion data.
    # If a bundle has inc_addtol or exc_addtol in its bundle_dict, that
    # tolerance will be added to this distance.
    # For example, if you wanted to increase tolerance for the right
    # arcuate waypoint ROIs by 3 each, you could make the following
    # modification to your bundle_dict:
    # bundle_dict["ARC_R"]["inc_addtol"] = [3, 3]
    # Additional tolerances can also be negative.
    dist_to_waypoint = ""
    
    # If None, creates RandomState.
    # If int, creates RandomState with seed rng.
    # Used in RecoBundles Algorithm.
    # Default: None.
    rng = ""
    
    # Whether to return the indices in the original streamlines as part
    # of the output of segmentation.
    return_idx = false
    
    # If not None, presegment by ROIs before performing
    # RecoBundles. Only used if seg_algo starts with 'Reco'.
    # Meta-data for the segmentation. The format is something like::
    # {'bundle_name': {
    # 'include':[img1, img2],
    # 'prob_map': img3,
    # 'cross_midline': False,
    # 'start': img4,
    # 'end': img5}}
    # Default: None
    presegment_bundle_dict = ""
    
    # Optional arguments for initializing the segmentation for the
    # presegmentation. Only used if presegment_bundle_dict is not None.
    # Default: {}
    presegment_kwargs = "{}"
    
    # Whether to filter the bundles based on their endpoints.
    # Applies only when `seg_algo == 'AFQ'`.
    # Default: True.
    filter_by_endpoints = true
    
    # If filter_by_endpoints is True, this is the required distance
    # from the endpoints to the atlas ROIs.
    dist_to_atlas = 4
    
    # The full path to a folder into which intermediate products
    # are saved. Default: None, means no saving of intermediates.
    save_intermediates = ""
    
    [CLEANING_PARAMS]
    
    # Number of points to resample streamlines to.
    # Default: 100
    n_points = 100
    
    # Number of rounds of cleaning based on the Mahalanobis distance from
    # the mean of extracted bundles. Default: 5
    clean_rounds = 5
    
    # Threshold of cleaning based on the Mahalanobis distance (the units are
    # standard deviations). Default: 3.
    distance_threshold = 3
    
    # Threshold for cleaning based on length (in standard deviations). Length
    # of any streamline should not be *more* than this number of stdevs from
    # the mean length.
    length_threshold = 4
    
    # Number of streamlines in a bundle under which we will
    # not bother with cleaning outliers. Default: 20.
    min_sl = 20
    
    # The statistic of each node relative to which the Mahalanobis is
    # calculated. Default: `np.mean` (but can also use median, etc.)
    stat = "mean"
    
    # Whether to return indices in the original streamlines.
    # Default: False.
    return_idx = false
    
    [DATA]
    
    # Minimum b value you want to use
    # from the dataset (other than b0), inclusive.
    # If None, there is no minimum limit. Default: None
    min_bval = ""
    
    # Maximum b value you want to use
    # from the dataset (other than b0), inclusive.
    # If None, there is no maximum limit. Default: None
    max_bval = ""
    
    # Whether to filter the DWI data based on min or max bvals.
    # Default: True
    filter_b = true
    
    # The value of b under which
    # it is considered to be b0. Default: 50.
    b0_threshold = 50
    
    # Whether to use robust_tensor_fitting when
    # doing dti. Only applies to dti.
    # Default: False
    robust_tensor_fitting = false
    
    # The response function to be used by CSD, as a tuple with two elements.
    # The first is the eigen-values as an (3,) ndarray and the second is
    # the signal value for the response function without diffusion-weighting
    # (i.e. S0). If not provided, auto_response will be used to calculate
    # these values.
    # Default: None
    csd_response = ""
    
    # default: infer the number of parameters from the number of data
    # volumes, but no larger than 8.
    # Default: None
    csd_sh_order = ""
    
    # weight given to the constrained-positivity regularization part of
    # the deconvolution equation. Default: 1
    csd_lambda_ = 1
    
    # threshold controlling the amplitude below which the corresponding
    # fODF is assumed to be zero.  Ideally, tau should be set to
    # zero. However, to improve the stability of the algorithm, tau is
    # set to tau*100 percent of the mean fODF amplitude (here, 10 percent
    # by default)
    # (see [1]_). Default: 0.1
    csd_tau = 0.1
    
    # The sphere providing sample directions for the initial
    # search of the maximal value of kurtosis.
    # Default: 'repulsion100'
    sphere = "repulsion100"
    
    # This input is to refine kurtosis maxima under the precision of
    # the directions sampled on the sphere class instance.
    # The gradient of the convergence procedure must be less than gtol
    # before successful termination.
    # If gtol is None, fiber direction is directly taken from the initial
    # sampled directions of the given sphere object.
    # Default: 1e-2
    gtol = 0.01
    
    # This will be used to create
    # the brain mask, which gets applied before registration to a
    # template.
    # If you want no brain mask to be applied, use FullImage.
    # If None, use B0Image()
    # Default: None
    brain_mask_definition = ""
    
    # List of bundle names to include in segmentation,
    # or a bundle dictionary (see BundleDict for inspiration),
    # or a BundleDict. See `Defining Custom Bundle Dictionaries`
    # in the `usage` section of pyAFQ's documentation for details.
    # If None, will get all appropriate bundles for the chosen
    # segmentation algorithm.
    # Default: None
    bundle_info = ""
    
    # The target image data for registration.
    # Can either be a Nifti1Image, a path to a Nifti1Image, or
    # if "mni_T2", "dti_fa_template", "hcp_atlas", or "mni_T1",
    # image data will be loaded automatically.
    # If "hcp_atlas" is used, slr registration will be used
    # and reg_subject should be "subject_sls".
    # Default: "mni_T1"
    reg_template_spec = "mni_T1"
    
    [MAPPING]
    
    # This defines how to either create a mapping from
    # each subject space to template space or load a mapping from
    # another software. If creating a map, will register reg_subject and
    # reg_template.
    # If None, use SynMap()
    # Default: None
    mapping_definition = ""
    
    # The source image data to be registered.
    # Can either be a Nifti1Image, an ImageFile, or str.
    # if "b0", "dti_fa_subject", "subject_sls", or "power_map,"
    # image data will be loaded automatically.
    # If "subject_sls" is used, slr registration will be used
    # and reg_template should be "hcp_atlas".
    # Default: "power_map"
    reg_subject_spec = "power_map"
    
    [SEGMENTATION]
    
    # How to weight each streamline (1D) or each node (2D)
    # when calculating the tract-profiles. If callable, this is a
    # function that calculates weights. If None, no weighting will
    # be applied. If "gauss", gaussian weights will be used.
    # If "median", the median of values at each node will be used
    # instead of a mean or weighted mean.
    # Default: "gauss"
    profile_weights = "gauss"
    
    # List of scalars to use.
    # Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md", "dki_awf",
    # "dki_mk". Can also be a scalar from AFQ.definitions.image.
    # Default: ["dti_fa", "dti_md"]
    scalars = "['dti_fa', 'dti_md']"
    
    [TRACTOGRAPHY]
    
    # BIDS filters for inputing a user made tractography file,
    # or a path to the tractography file. If None, DIPY is used
    # to generate the tractography.
    # Default: None
    import_tract = ""
    
    [VIZ]
    
    # Of the form (lower bound, upper bound). Shading based on
    # shade_by_volume will only differentiate values within these bounds.
    # If lower bound is None, will default to 0.
    # If upper bound is None, will default to the maximum value in
    # shade_by_volume.
    # Default: [None, None]
    sbv_lims_bundles = "[None, None]"
    
    # Opacity of volume slices.
    # Default: 0.3
    volume_opacity_bundles = 0.3
    
    # n_points to resample streamlines to before plotting. If None, no
    # resampling is done.
    # Default: 40
    n_points_bundles = 40
    
    # Of the form (lower bound, upper bound). Shading based on
    # shade_by_volume will only differentiate values within these bounds.
    # If lower bound is None, will default to 0.
    # If upper bound is None, will default to the maximum value in
    # shade_by_volume.
    # Default: [None, None]
    sbv_lims_indiv = "[None, None]"
    
    # Opacity of volume slices.
    # Default: 0.3
    volume_opacity_indiv = 0.3
    
    # n_points to resample streamlines to before plotting. If None, no
    # resampling is done.
    # Default: 40
    n_points_indiv = 40
    
    # Which visualization backend to use.
    # See Visualization Backends page in documentation for details:
    # https://yeatmanlab.github.io/pyAFQ/usage/viz_backend.html
    # One of {"fury", "plotly", "plotly_no_gif"}.
    # Default: "plotly_no_gif"
    viz_backend_spec = "plotly_no_gif"
    
    # Whether to use a virtual fram buffer. This is neccessary if
    # generating GIFs in a headless environment. Default: False
    virtual_frame_buffer = false
    
    
    
pyAFQ will store a copy of the configuration file alongside the computed
results. Note that the `title` variable and `[metadata]` section are both for
users to enter any title/metadata they would like and pyAFQ will generally
ignore them.
