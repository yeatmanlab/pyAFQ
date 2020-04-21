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


