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


