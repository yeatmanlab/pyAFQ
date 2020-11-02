0.5 (October 13, 2020)
======================
This release allow users to provide custom tractography and registration
templates using pyBIDS filters.
We added the new tractography method PFT/ACT and the new reconstruction method
MSMT. RecoBundles can now use the Yeh 80 bundle atlas. Many minor bug fixes
and enhancements.
  * Save non anat positioned bundles in their own plots (#539)
  * Allow bundle dict as input to afq object (#540)
  * Put msmt in nightly 3 (#542)
  * Actually use MSMT in API call (#530)
  * Update test_init to additional df columns; add nightly 5 test yml (#531)
  * NIGHTLY: move reco80 to 4; break up nightly 3 to isolate bug (#525)
  * Multi-shell, multi-tissue model (#474)
  * BF: Reset the input tractogram space to what you originally got. (#520)
  * FIX: Remove xvfb from being needed in tests (#522)
  * FIX: Reco80 fixes (#503)
  * FIX: specify that the b val range is inclusive (#523)
  * ENH #443 Callosal Group Example (#476)
  * adding separate example output directories to avoid name collisions (#490)
  * FIX: Plotly viz bug fixes, and update to custom bundles (#513)
  * FIX: Update all b0_thresholds to 50 (#507)
  * FIX: update reg_subject arg checking to include dicts (#515)
  * FIX: updates docstring in clean_bundle which returns sft, not nib.streamlines (#514)
  * Try plotly as default, re-organize usage documentation (#439)
  * FIX: Update tractography max_length docstring to be consistent (#508)
  * Split off nightly 4 from nightly 3, nightly rests run python 3.8 (#501)
  * ENH: Sort the bundles list from csv alphabetically in group csv comparison (#499)
  * ENH Disabling progress bars for sphinx-gallery (#492)
  * ENH: Dice coeff (#484)
  * enh adding distclean and realclean targets for sphinx build (#489)
  * FIX: Downsample number of streamlines vizzed down to 200 when vizzing all bundles (#482)
  * ENH: Return contast index dataframe from contrast_index (#483)
  * Require dipy versions higher than 0.12, so that we can use current ma… (#488)
  * Reg algo automatically chosen based on whether mapping is provided, syn mapping for sls fixed, recobundle defaults updated (#472)
  * Apply brain mask to subject img before registration (#478)
  * FIX: export registered b0 should use inverse pre align to read mapping file (#479)
  * Fix typos in api.afq type checking (#477)
  * Type check AFQ object inputs (#456)
  * Allow reg subject to be bids filters; refactor getting stanford data into temp folder (#458)
  * Removes extraneous underscore. (#475)
  * Adds total number of streamlines, to complete the sl counts table. (#469)
  * Follow up on #462, to fix failing nightly test. (#470)
  * FIX: remove whole brain from bundle list for sl count when using recobundles (#471)
  * Fixes the timing table (#467)
  * Updating pyAFQ documentation (#455)
  * Streamline counts table (#468)
  * Adjusting group comparison figures (#466)
  * Adds 80-bundle atlas for RB (#431)
  * Catch no subject error early (#463)
  * Colorful legend when using cbv (#465)
  * Give the APM map a more BIDS-ish name, and generate metadata file. (#462)
  * FIX: Validate bids_path exists (#459)
  * Give users the option to disable individual subject lines in plot (#446)
  * Add PFT Mask. (#444)
  * Rename profile columns back to standard column names (#445)
  * Update plots to paper Quality (#429)
  * Remove whole brain from bundle dict for reco viz (#438)
  * Allow option to upload combined profile (#437)
  * Add packaging requirement (#436)
  * Allow user to input custom tractography (#422)
  * Automatically choose bundle names for recobundles (#420)
  * Skip nightly test that may be causing OOM (#428)


0.4.1 (September 14, 2020)
========================

This micro release provides testing for Python 3.8 and multiple other fixes.
This release requires the newly-released DIPY 1.2.0

  * BF: Register the PVE to the parameters. (#423)
  * Python 3.8 (#360)
  * Further subdivide nightly tests (#419)
  * Many minor bug fixes (#409)
  * Divide nightly test into 2; have nightly tests only do nightly tests (#417)
  * Raise error earlier for empty ROI (#415)
  * Update example to serve as a CI target (#380)
  * Convert local_directories set to list (#414)
  * Update to Dipy 1.2.0 (#384)
  * Adds ParticleFilteringTractography. (#335)
  * A few visualization updates (#390)
  * Timing dict name mismatch bug fix (#395)
  * make decision to combine profile automatic in export_all (#387)
  * Export timing information (#393)
  * Remove unnecessary check (#389)
  * Add ability to remove edges of profiles (#386)
  * Add upload_to_s3, download_and_combine_afq_profiles (#376)
  * Change nighty tests to happen at midnight, PST (#383)
  * Use get_fdata instead of deprecated get_data in example (#377)
  * Skip seg faulting test (#378)
  * Overhaul Mask File UI; Other minor UI improvements (#370)
  * Return a unique set of subjects in S3BIDSStudy._all_subjects (#373)
  * Allow user to seed tractography with ROIs via api. Use this to reduce test times. Mark some tests as nightly. (#364)
  * Added test that runs full cli pipeline on default config file. Tell CI to not run the tests marked slow. (#356)


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


