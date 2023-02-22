1.0.1 (February 22, 2023)
=========================
Fixed two bugs, where max streamline length was not being enforced and the
mean b0 was being calculated incorrectly. Small documenation fixes.
  * [FIX] Add max length constraint, b0 fix (#955)
  * Small documentation fixes, to improve auto-rendering of API docs. (#949)
  * MAINT: Upgrade pytest as a start to tackling CI errors. (#945)
  * MAINT: pep517 => build. (#947)

1.0 (January 05, 2023)
======================
This is the first major release of pyAFQ. The API should be stable
from here until the next major release of pyAFQ.
Adds freewater DTI, fiber density maps, VOF and pAF bundles,
and montage and combine bundle visualizations, as well as other
minor additions, fixes, and documentation updates.
  * [FIX] Missing underscore in custom tissue property filenames (#944)
  * [FIX] finally finish this SLR PR (#937)
  * Add fwDTI (#931)
  * [FIX] Import tract docs (#936)
  * TST: Fixes another failing nightly test by setting the seed. (#932)
  * DOC: Fix the random seed for the OR example. (#930)
  * TST: Fix a random seed for this test. (#929)
  * [FIX] don't resample subject-space ROIs unless user provides something (#919)
  * [ENH] Density task (#900)
  * Increase inclusion tolerance for pAF; add suffix for cmd_outputs (#921)
  * [DOC] add interactive bundle viz to OR example (#861)
  * [ENH] Allow pediatric bundledict and template to be accessed from config file. (#923)
  * [FIX] Some pediatric bundle dict fixes (#922)
  * DOC: Adds an example of visualizations using FURY. (#910)
  * [FIX] put afq_metadata.toml in the correct spot (#913)
  * [FIX] Updating DIPY tracking params (#911)
  * [ENH] VOF endpoints don't include pAF endpoints (#912)
  * DOC: Document S3BIDS access examples. (#909)
  * [FIX] correctly calculate min / max sl length, update step_size docs (#905)
  * Added streamline filtering by primary orientation; other bundle definition fixes (#898)
  * [FIX] cmd output only run on our files (#881)
  * Set logger levels to INFO (#867)
  * [DOC] add pyafq overview desciption (#875)
  * Fix nightly tests (#873)
  * Upgrade pybids. (#869)
  * [ENH] Apply arbitrary command to some/all pyAFQ outputs, more BIDSy names (#853)
  * [FIX] only calc subject registration images when necessary (#862)
  * RF: Removes key-word argument that is not being used. (#868)
  * ENH: add DKI kurtosis fit elements as scalars (#865)
  * ENH: add lower triangular scalars from DTI (could be useful for ML)? (#863)
  * FIX: Reorder endpoint (#858)
  * [FIX] Some ITK map fixes for running with groupAFQ and in CLI (#855)
  * [ENH] Add montage and combine bundle to GroupAFQ (#849)
  * [DOC] add fatal error when no bundles found at all (#851)
  * [ENH] add median bundle len function (#852)

0.12.2 (July 21, 2022)
======================
This release fixes a critical bug introduced in version 0.11, which
caused exclusion ROIs to be ignored.
  * [FIX]: Reorder endpoint (#858)
  * [FIX] Some ITK map fixes for running with groupAFQ and in CLI (#855)
  * [ENH] Add montage and combine bundle to GroupAFQ (#849)
  * [DOC] add fatal error when no bundles found at all (#851)
  * [ENH] add median bundle len function (#852)


0.12.1 (June 30, 2022)
======================
More informative warnings and errors, minor fixes.
  * [FIX] warn user about number of invalid streamlines removed (#850)
  * TST: Test the nibabel release candidate. (#842)
  * MAINT: Refreshes the zenodo metadata file. (#845)
  * [ENH] minor docs updates and qsiprep changes (#847)
  * [FIX] Error when file not found should be more informative (#844)
  * [ENH] Return path to single viz file if its generated (#839)
  * [ENH] Add core_bundle functionality to fury backend (#838)
  * [FIX] update to_call in pyAFQ CLI to new API (#836)

0.12 (May 18, 2022)
===================
This release simplifies the API, in part by merging the
scalar and mask definitions into one image definition.
pyAFQ now must use python >=3.8. Other minor bug fixes.
  * [FIX] update nightly tests and fix kwargs (#834)
  * [DOC] update track docstring (#833)
  * [ENH] Ignore pandas out of date warning / SFT reference warning (#832)
  * [ENH] Simplify API system (#825)
  * [FIX] always resample when loading from disk, not just first time (#830)
  * [FIX] fix ImageFile for scalars, add test (#829)
  * [ENH] replace parfor with paramap (#828)
  * [FIX] Replace split(.) with splitext() (#827)
  * [ENH] Change pyAFQ to use python >=3.8 (#826)
  * [ENH] Autogenerate two separate QSIprep pipelines (#816)
  * [ENH] Reorganize mask and scalar system into one "Image" system (#817)
  * Update DIPY to 1.5 (#814)
  * [ENH] Allow user to customize core bundle text indicating nodeID (#815)

0.11 (April 12, 2022)
=====================
This release introduces a new API for specifying Bundle Dictionaries,
which should make it more straightforward to create custom bundle
dictionaries. In addition, there are a few minor enhancements and updates
to the documentation.
  * Reduce number of streamlines in nightly reco80 test (#813)
  * Reduce memory consumption of Reco80 test (#812)
  * Upgrade moto (#811)
  * DOC: Add pointer to discussions page in getting_help.rst (#809)
  * Split this nightly test into two separate nightly tests. (#807)
  * [test] Move reco80 from custom to anisotropic nightly test (#805)
  * [ENH] Allow segmentation tie breakers to be settled by distance from ROI (#804)
  * Remove MSMT from pyAFQ (#803)
  *  (#801)
  * [DOC] update scalars documentation & split API methods description to its own page (#796)
  * [FIX] Some fixes for nightly tests (#794)
  * [FIX] update cvxpy (#793)
  * DOC: Adds intersphinx mapping to numpy python dipy (#230)
  * [ENH] Better BundleDict System (#788)
  * [ENH] Updated model defaults (#792)
  * [DOC] fix minor erros in documenation (#786)
  * Adds CITATION file. (#787)
  * [FIX,ENH] fix typo in docs, bug in GroupAFQ init, add export_all to ParticipantAFQ (#784)
  * [FIX] use plotly cmap instead of matplotlib cmap in plotly_backend (#785)

0.10 (December 07, 2021)
========================
This release introduces a new API for BIDS-organized group studies
(`GroupAFQ`). This API is backwards-compatible (apart from the name) with
the API of the now-deprecated `AFQ` class. A new class is introduced that
provides more flexibility for users: `ParticipantAFQ`, which accepts
data for a single participant in any format (i.e., non-BIDS), so long
as strings pointing to the full paths of the data, bvals, bvecs can
be provided.

  * NF: Handle situations where CSD auto-response function cannot complete. (#776)
  * Group AFQ / Participant AFQ (#764)
  * [ENH] allow user to pass arguments to pyBIDS BIDSLayout (#774)


0.9.2 + 0.9.3 (November 18, 2021)
==================================
These bug fix releases are meant to improve tagging of Docker images.

  * [ENH] try to get the tag name even not on release (#775)


0.9.1 (November 12, 2021)
=========================
This is a bug-fix release, which fixes some issues with the CLI in the previous
release. It also implements a docker entrypoint and should improve automated
tagging of docker images.

  * [FIX] Remember to add docker-push.sh (#772)
  * [ENH] Add entrypoint for pyAFQ docker image (#769)
  * [ENH] Update the docker worklow to hopefully push tags, as well as remove AFQsi docker (#770)
  * [FIX] this is a quick fix for the problem with using the default config file (#768)
  * [ENH] Identity Map (#758)
  * [ENH] remove patch2self (#757)
  * Suppress warnings when using api (#759)


0.9 (October 25, 2021)
======================
This is a maintenance release, including many small fixes to specific
issues that arose during usage with particular datasets. It also includes
some improvements to visualizations. This version includes some of the
requirements for a BIDS App: participant list and output directory and
the initial requirements for integration with QSIPrep.

  * Generate json for QSIprep from command line (#744)
  * Enh: rename this variable (thoughts?) (#756)
  * Enh: Reduce pyAFQ required dependencies (#752)
  * [ENH] Change default BundleDict behavior to resample (#755)
  * [ENH] alert users when custom tractography is not found for a given sub/ses (#754)
  * [FIX] Clean up Loky (#750)
  * [FIX] Attempt to fix the docker push for tags (#751)
  * [ENH] Participant labels implementation (#749)
  * [FIX] fury nightly fix (#748)
  * Fixes a typo in this variable name. (#747)
  * Allow other extensions than nii.gz to be given by the user for optional input files (#745)
  * [ENH] Replicating mAFQ visualizations using our rendering framework (#736)
  * BF: Resample ROI for custom bundledict as well (#742)
  * pyBabyAFQ (#524)
  * [ENH] Allow AFQ browser installation to be optional (#740)
  * Updates qsiprep version to inherit from. (#741)
  * [BF] ITK and FNIRT mappings had typo reversing reg_subject and reg_template (#739)
  * [DOC] Generate simple docs for export function outputs (#729)
  * [DOC] Adds an example to the custom scalar docs (#732)
  * Adding a citation file. (#734)
  * [ENH] add more scalars, add one to the test (#733)
  * DOC: Insert the current version into the documentation. (#731)


0.8 (July 12, 2021)
===================
This release is the first one to use Pimms as our pipeline engine, which allows
for parallelization across subjects and sessions using multi-processing. It also
contains integration of AFQ-Browser as a derivative, and a variety of other
fixes and improvments.

  * WIP: Add OR fetcher and example (#646)
  * [Fix] Better export all behavior (#726)
  * [TESTFIX] Update nightly test to match new, more specific error message (#727)
  * BF: A couple of places where masks are not being propagated. (#721)
  * [FIX] Default to serial subject-session processing to ease memory constraints (#720)
  * [FIX] fix custom bundldict bug and add test (#718)
  * [DOC] Add developer documentation for adding tasks and definitions (#714)
  * BF: Fix config file reader for new params (#713)
  * BF: loop over valid_sub and valid_ses lists correctly (#712)
  * Fixes broken link in BIDS example  (#709)
  * Move AAL atlas to figshare (#710)
  * BF: Fix docker builds  (#708)
  * [ENH] AFQ-Browser Integration (#703)
  * [DOC] Bids layout clarifications (#697)
  * [ENH] Provide more informative errors for incorrect BIDS structure, generate bundle dict lazily (#691)
  * [FIX] Mask getters have to be aware of whether they are being called from data task (#705)
  * [Nightly] Split nightly basic again (#706)
  * [Nightly] Rename nightly tests, split test 2 (#704)
  * [FIX] Nightly pimms fixes2 (#701)
  * BF: Only generate warped endpoint ROIs if there are endpoint ROIs to use (#700)
  * [FIX] Nightly pimms fixes (#699)
  * [FIX] allow for session folder to not exist (session to be None) (#694)
  * Pimms (#675)
  * [ENH] Use ICC for profile reliability (#690)
  * Allow for other derivatives folders when downloading and combining profiles (#689)
  * Fixes link on front page (#687)


0.7.1 (May 03, 2021)
====================
This micro release improves the look and feel of the documentation.
Also, provides tagged docker images.

  * [RF] Builds a tagged image when a tag is pushed. (#677)
  * [DOC] Update docs to clarify where pipeline name comes from (#686)
  * [FIX] download and combine profile fixes and improvements (#685)
  * [FIX] cloudknot example fixes (#682)
  * [WIP] [FIX] Attempt to get doc examples to run again (#683)
  * [DOC]: Overhauls docs front page. (#673)
  * [BF]: Update s3fs version. (#678)
  * [FIX] More lenient reco defaults (#657)


0.7 (April 06, 2021)
====================
This version includes upgrades and updates to a variety of methods.
A major new feature in this release is automated builds of docker images for
both pyAFQ by itself, as well as in tandem with qsiprep.

* [FIX] Minor BF for nightly tests (#665)
  * [ENH] A variety of registration improvements, primarily for babyAFQ (#661)
  * [Doc] try new docs layout (#664)
  * [FIX] Build and push pyAFQ docker image only after merge (#663)
  * [ENH] docker workflow for pyafq and afqsi (#659)
  * only run roi dilation if necessary (#662)
  * BF: Set up bundle dict in cases where a dict is provided, but algo is "reco" (#658)
  *  [FIX] try to make pyafq play nice with pybids 0.9.3 (#660)
  * BF: _gen_sl_counts function was failing with more than one subject (#656)
  * [FIX] remove invalid sls from tractography, which could be custom (#654)
  * [FIX] Propogate flip axial to export_all (#651)
  * [ENH] make cvxpy optional (#653)
  * Allow ItkMap in pyAFQ (#650)
  * Dipy should be at least 1.4.0 (#643)
  * Median tract profile (#649)
  * Some minor bug fixes/improvements from the optic bids PR (#647)
  * Major changes for processing optic radiations with CLI (#625)
  * [ENH] Input ylim for CI plots (#642)
  * [FIX] Plotly Bundle Visualization fixes (#641)
  * DOC: This page has since moved. (#640)
  * DOC: When releasing, we need to push the tag upstream. (#639)
  * Add API method to export masked b0 (#638)
  * [FIX] opacity argument to make fury API like plotly API (#637)
  * Add conflict checker; loosen up dependencies (#636)
  * Allow more flexibility in dask version (#634)
  * More plotting tweaks, gridspec compat (#627)
  * Added an ROI pre segmentation as an option for recobundles (#573)
  * Relax versions to avoid cvxpy/numpy incompatibility (#632)
  * export endpoint ROI when saving intermediates (#628)
  * FIX: combine AFQ profiles (#585)
  * Fixes DCP Error (#630)
  * Update numpy version (#629)
  * Allow user to only use prealign in registration (#626)


0.6 (January 05, 2021)
======================
This version includes many fixes, documentation enhancements and
performance optimizations. It also drops Python 3.6 support.
This version matches our first paper describing/using the software.

  * Add more timing information (#622)
  * Allow CLI to specify what api method is called (#623)
  * Plot tweaks to make paper quality (#576)
  * Reduce apm test workload (#621)
  * Split up nightly 5 (#620)
  * Fix test_AFQ_init, may allow nightly 3 to pass (#619)
  * Dilate the ROIs. (#618)
  * Enh Add Callosum ROIs support (#538)
  * BF: Need to read these parameters from file, before getting the data. (#615)
  * Drop Python 3.6 support. (#612)
  * BF: use get instead of get_nearest (#610)
  * [ENH] [DOC] Add matlab to python file conversion functions, add docs for custom tractography integration (#599)
  * [FIX] calculate sl counts on the spot (#605)
  * DOC: Example that explores BIDS and includes tractography from another pipeline. (#577)
  * Allow more flexible definition of endpoint filtering atlas. (#589)
  * DOC: Explain that trk files are saved in RASMM. (#604)
  * Removes several unused CLIs. (#588)
  * BF: Use the provided x and y inputs. (#606)
  * [ENH] added reco80 example (#567)
  * [DOC]Add mask.rst file to give context and explanation to masks (#598)
  * Reuse the CSD sh coefficients if you already have them. (#591)
  * [ENH] Allow the user to specify what range the color by volume should shade over (#594)
  * Adding dummy end points for custom bundles (#543)
  * [ENH] Allow user to optionally normalize density map maximum values to 1 (#595)
  * [ENH] Add cloudknot example (#533)
  * [ENH] Robust tensor fitting (#580)
  * FIX: Traverse BIDS hierarchy to find masks, bvals, and bvecs (#587)
  * NF: Adds DKI AWF to scalars. (#592)
  * Read and resample ROI data. (#545)
  * DOC: Adds some documentation for developers of the software. (#546)
  * initialize for subject and session pairs where dwi files exist (#583)
  * [FIX] Put tg in rasmm first for SLR registration (#566)
  * [FIX] Unflip Plotly x axis; multiple flexibility improvements in plotly plotting for paper (#581)
  * WIP DEP: Bump numpy version (#579)
  * adding nb_streamlines to segmentation parameters (#570)
  * [ENH] add weighted option for dice (#568)
  * Adds prealign stage to the examples. (#555)
  * Eliminate `force_recompute` option. (#552)
  * Warn when templateflow creates directory (#557)
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
