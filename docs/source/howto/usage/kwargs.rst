
.. _kwargs_docs:

The pyAFQ API optional arguments
--------------------------------
You can run pyAFQ on either a subject or participant level
using pyAFQ's API objects, :class:`AFQ.api.group.GroupAFQ`
and :class:`AFQ.api.participant.ParticipantAFQ`. Either way,
these classes take additional optional arguments. These arguments
give the user control over each step of the tractometry pipeline,
allowing customizaiton of tractography, bundle recognition, registration,
etc. Here are all of these arguments and their descriptions, organized
into 5 sections:

Here are the arguments you can pass to kwargs, to customize the tractometry pipeline. They are organized into 5 sections.

==========================================================
DATA
==========================================================
min_bval: float, optional
	Minimum b value you want to use
	from the dataset (other than b0), inclusive.
	If None, there is no minimum limit. Default: None

max_bval: float, optional
	Maximum b value you want to use
	from the dataset (other than b0), inclusive.
	If None, there is no maximum limit. Default: None

filter_b: bool, optional
	Whether to filter the DWI data based on min or max bvals.
	Default: True

b0_threshold: int, optional
	The value of b under which
	it is considered to be b0. Default: 50.

robust_tensor_fitting: bool, optional
	Whether to use robust_tensor_fitting when
	doing dti. Only applies to dti.
	Default: False

csd_response: tuple or None, optional.
	The response function to be used by CSD, as a tuple with two elements.
	The first is the eigen-values as an (3,) ndarray and the second is
	the signal value for the response function without diffusion-weighting
	(i.e. S0). If not provided, auto_response will be used to calculate
	these values.
	Default: None

csd_sh_order: int or None, optional.
	default: infer the number of parameters from the number of data
	volumes, but no larger than 8.
	Default: None

csd_lambda_: float, optional.
	weight given to the constrained-positivity regularization part of
	the deconvolution equation. Default: 1

csd_tau: float, optional.
	threshold controlling the amplitude below which the corresponding
	fODF is assumed to be zero.  Ideally, tau should be set to
	zero. However, to improve the stability of the algorithm, tau is
	set to tau*100 percent of the mean fODF amplitude (here, 10 percent
	by default)
	(see [1]_). Default: 0.1

sphere: Sphere class instance, optional
	The sphere providing sample directions for the initial
	search of the maximal value of kurtosis.
	Default: 'repulsion100'

gtol: float, optional
	This input is to refine kurtosis maxima under the precision of
	the directions sampled on the sphere class instance.
	The gradient of the convergence procedure must be less than gtol
	before successful termination.
	If gtol is None, fiber direction is directly taken from the initial
	sampled directions of the given sphere object.
	Default: 1e-2

brain_mask_definition: instance from `AFQ.definitions.image`, optional
	This will be used to create
	the brain mask, which gets applied before registration to a
	template.
	If you want no brain mask to be applied, use FullImage.
	If None, use B0Image()
	Default: None

bundle_info: strings, dict, or BundleDict, optional
	List of bundle names to include in segmentation,
	or a bundle dictionary (see BundleDict for inspiration),
	or a BundleDict. See `Defining Custom Bundle Dictionaries`
	in the `usage` section of pyAFQ's documentation for details.
	If None, will get all appropriate bundles for the chosen
	segmentation algorithm.
	Default: None

reg_template_spec: str, or Nifti1Image, optional
	The target image data for registration.
	Can either be a Nifti1Image, a path to a Nifti1Image, or
	if "mni_T2", "dti_fa_template", "hcp_atlas", or "mni_T1",
	image data will be loaded automatically.
	If "hcp_atlas" is used, slr registration will be used
	and reg_subject should be "subject_sls".
	Default: "mni_T1"


==========================================================
MAPPING
==========================================================
mapping_definition: instance of `AFQ.definitions.mapping`, optional
	This defines how to either create a mapping from
	each subject space to template space or load a mapping from
	another software. If creating a map, will register reg_subject and
	reg_template.
	If None, use SynMap()
	Default: None

reg_subject_spec: str, instance of `AFQ.definitions.ImageDefinition`, optional  # noqa
	The source image data to be registered.
	Can either be a Nifti1Image, an ImageFile, or str.
	if "b0", "dti_fa_subject", "subject_sls", or "power_map,"
	image data will be loaded automatically.
	If "subject_sls" is used, slr registration will be used
	and reg_template should be "hcp_atlas".
	Default: "power_map"


==========================================================
SEGMENTATION
==========================================================
segmentation_params: dict, optional
	The parameters for segmentation.
	Default: use the default behavior of the seg.Segmentation object.

clean_params: dict, optional
	The parameters for cleaning.
	Default: use the default behavior of the seg.clean_bundle
	function.

profile_weights: str, 1D array, 2D array callable, optional
	How to weight each streamline (1D) or each node (2D)
	when calculating the tract-profiles. If callable, this is a
	function that calculates weights. If None, no weighting will
	be applied. If "gauss", gaussian weights will be used.
	If "median", the median of values at each node will be used
	instead of a mean or weighted mean.
	Default: "gauss"

scalars: strings and/or scalar definitions, optional
	List of scalars to use.
	Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md", "dki_awf",
	"dki_mk". Can also be a scalar from AFQ.definitions.image.
	Default: ["dti_fa", "dti_md"]


==========================================================
TRACTOGRAPHY
==========================================================
tracking_params: dict, optional
	The parameters for tracking. Default: use the default behavior of
	the aft.track function. Seed mask and seed threshold, if not
	specified, are replaced with scalar masks from scalar[0]
	thresholded to 0.2. The ``seed_mask`` and ``stop_mask`` items of
	this dict may be ``AFQ.definitions.image.ImageFile`` instances.
	If ``tracker`` is set to "pft" then ``stop_mask`` should be
	an instance of ``AFQ.definitions.image.PFTImage``.

import_tract: dict or str or None, optional
	BIDS filters for inputing a user made tractography file,
	or a path to the tractography file. If None, DIPY is used
	to generate the tractography.
	Default: None


==========================================================
VIZ
==========================================================
sbv_lims_bundles: ndarray
	Of the form (lower bound, upper bound). Shading based on
	shade_by_volume will only differentiate values within these bounds.
	If lower bound is None, will default to 0.
	If upper bound is None, will default to the maximum value in
	shade_by_volume.
	Default: [None, None]

volume_opacity_bundles: float, optional
	Opacity of volume slices.
	Default: 0.3

n_points_bundles: int or None
	n_points to resample streamlines to before plotting. If None, no
	resampling is done.
	Default: 40

sbv_lims_indiv: ndarray
	Of the form (lower bound, upper bound). Shading based on
	shade_by_volume will only differentiate values within these bounds.
	If lower bound is None, will default to 0.
	If upper bound is None, will default to the maximum value in
	shade_by_volume.
	Default: [None, None]

volume_opacity_indiv: float, optional
	Opacity of volume slices.
	Default: 0.3

n_points_indiv: int or None
	n_points to resample streamlines to before plotting. If None, no
	resampling is done.
	Default: 40

viz_backend_spec: str, optional
	Which visualization backend to use.
	See Visualization Backends page in documentation for details:
	https://yeatmanlab.github.io/pyAFQ/usage/viz_backend.html
	One of {"fury", "plotly", "plotly_no_gif"}.
	Default: "plotly_no_gif"

virtual_frame_buffer: bool, optional
	Whether to use a virtual fram buffer. This is neccessary if
	generating GIFs in a headless environment. Default: False

