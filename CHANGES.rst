0.4 (August 17, 2020)
=====================
This release adds several new registration templates and techniques, providing 
major improvements in bundle segmentation with waypoint ROIs. In addition, new 
visualization methods using plotly were introduced, to generate HTML-based 
visualizations of bundles, and to compare longitudinal measurements. This version
relies on pybids for parsing of input datasets. Many other bug fixes and improvements.

  * Throw error when backend is misnamed (#363)
  * Export what is natural to the viz library (#361)
  * Upgrade FURY to support newer VTK (#359)
  * Allow for selecting subject(s) by position after randomization (#352)
  * Ignore positional arguments in non AFQ functions for docstring parser, add test (#354)
  * Auto doc fix (#350)
  * Clips streamlines by the inclusion ROI. (#159)
  * S3 Bids Fetch Fixes (#340)
  * BF: AFQ derivatives should be saved inside "derivatives/afq" (#348)
  * Compare profiles from CSVs (#317)
  * New CLI / Config (#310)
  * Update versions of scipy and vtk. (#342)
  * Generating a wm mask out of multiple scalars (#330)
  * Add classes for fetching a BIDS-compliant study on S3 (#290)
  * Remove this section of the pyproject. (#337)
  * Setup with config redux ... redux (#326)
  * Updates DIPY url used in metadata. (#333)
  * b0_threshold default updated (#331)
  * Make power maps the default for registration (#329)
  * b value selection fix and test, power map test, models moved to own submodule (#322)
  * Revert "Adds a setup.cfg file and cleans up metadata and other hacks" (#324)
  * Adds a setup.cfg file and cleans up metadata and other hacks (#320)
  * Allow user to customize registration targets, fix some other aspects of registration (#283)
  * Bvals preallocation (#63)
  * Remove hcc from defaults (#315)
  * export_all function (#312)
  * Kaleido instead of orcas (#311)
  * Viz module uses fury and plotly (#289)
  * Allow study selection for fetch_hcp (#300)
  * RF: Speed up testing by moving viz test into API run. (#302)
  * Pybids (#284)
  * Plot tract profiles in CLI (#280)
  * Remove cloukdnot examples (#297)
  * Make save intermediates folder if it does not already exist (#296)
  * Remove six (#295)
  * use rapidfuzz instead of fuzzywuzzy (#266)
  * Extra requirements feature added, consistent with current requirement files (#291)
  * Bump pybids to 0.11.1 (#293)
  * make_bundle_dict should only be called after reg_template is settled (#281)
  * Add instructions for disabling github actions on forked repos. (#287)
  * Fix two bugs, makes tests work (#282)
  * Visualize all tract profiles for a scalar in one plot (#268)
  * Profile format changed to be less tall and more wide, like mAFQ (#279)
  * Added random_seed parameter in tractography (#270)
  * Bring fetch_hcp into alignment with other fetch functions (#272)
  * Api File Naming (#269)
  * Fixed numpydoc version to 0.9.2 (#271)
  * Use xvfb_wrapper for aws (#257)
  * Save mask as float32, so that we can open it in MI-Brain. (#260)
  * Update reqs 253 (#254)
  * Use valid value for Zenodo license field. (#249)
  * BF: Use Tableau 20 colors for the 20 waypoint-defined bundles. (#251)
  * BF: Actually use the user-provided path for saving HCP files. (#250)


0.3 (April 21, 2020)
====================
This release includes several important fixes and enhancements. In particular,
prealignment of the template . Additional accuracy of alignment is provided by
registering to a skull-stripped template provided through `templateflow`. A new
class for fiber groups and bundles was introduced and animated visualizations of
results are provided as a standard part of the CLI pipeline.

  * NF: Add dki to api (#238, JK)
  * DOC: Fixes error in installation instructions (#244, AR)
  * NF: Add fiber group class (#184, JK)
  * RF: Read the MNI template from templateflow, instead of DIPY. (#224, AR)
  * NF: Adds a flag to save intermediate variables within Segmentation (#237, AR)
  * NF: Integrate the bundle gif viz into the CLI. (#242, AR)
  * NF: Automatically infer sh_order (#240, AR)
  * NF: Update bundle visualization and add export gif to API (#229, JK)
  * BF: Fix warping (#232, AN)
  * BF: More API Fixes (#228, JK)
  * BF: Restore force recompute (#225, AR)


0.2 (February 20, 2020)
=====================
This release introduces some changes to the main API module, to incorporate all
helper functions into the `API` class. This should not affect user API.
  * RF: Api Usability (#207; JK)


0.1.2 (February 14, 2020)
=========================
This release fixes a bug in the way that indices of streamlines are handled,
introduce a Boutiques descriptor and includes other small fixes for automation.

  * NF: CLI fixes and creation of a Boutiques descriptor (#206; GK)
  * NF: Usage tracking with popylar (#205; AR)
  * BF: `clean_by_endpoints` should be able to return the indices. (#204; AR)
  * DOC: Adds an example of downloading some example data and organizing it. (#211)
  * MAINT: Use only the setuptools_scm version string. Removing all others. (#199; AR)
  * MAINT: Use setuptools_scm to add the git hash to the version string. (#198; AR)
  * MAINT: Maintenance tools (#222; AR)


