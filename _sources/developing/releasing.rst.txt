How to create a release
=======================

The current release process entails the following steps:

#. Make sure you have the `master` branch locally updated to the state on the `main fork of the project <https://github.com/yeatmanlab/pyAFQ/tree/master>`_.

#. Update CHANGES.rst: run the script `.maintenance/update_changes.sh <next_tag>`, with the tag set to the version string of the upcoming release. Further edit the notes to add a short summary at the top.

#. Make a commit with these changes and push the commit to the main fork on GitHub (i.e., `upstream`).

#. Tag the release. For example, to release Version 0.5, we did::

    git tag -a 0.5 -m "Version 0.5"

#. Push the tag::

    git push upstream 0.5

#. On GitHub:
    - Navigate to the `"Releases" tab <https://github.com/yeatmanlab/pyAFQ/releases>`_
    - Click on `"Draft a new release" <https://github.com/yeatmanlab/pyAFQ/releases/new>`_.
    - In the "Tag version" box enter the version string of this release.
    - In "Release title" enter (for example) "Version 0.5".
    - In the description box copy the entry in CHANGES.rst corresponding to this release (with the exclusion of the title).

#. Announce the release through all channels currently used:

    - Slack to DIRECT collaborators.
    - More to come...
