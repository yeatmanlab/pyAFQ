How to develop `pyAFQ`
==================

We are glad that you are here! That probably means that you are interested in contributing to the development of `pyAFQ`.
Before you continue reading specific topics below, please read the contribution guidelines
`here <https://github.com/yeatmanlab/pyAFQ/blob/master/.github/CONTRIBUTING.md>`_ .

As an overview, the pyAFQ code can roughly be divided into 5 sections.

1 - Code which contains purely low-level functionality. This includes
    the viz folder, the data folder, registration.py ,
    tractography.py, and segmentation.py.
    As these sections become stable, their code may ultimately be
    transfered to DIPY.

2 - Code which contains the user facing API classes. This includes
    the API folder. Any functionality that applies to an entire dataset
    is either in these files or called to from these files.

3 - Code which describes tasks within the tractometry workflow.
    **For most contributors, this is a good place to start**. The user-facing API/CLI
    use a workflow automatically constructed from these discrete tasks,
    as described in `tasks <https://yeatmanlab.github.io/pyAFQ/developing/tasks.html>`_ .
    For shorter tasks, one may only need to edit the relevant task file.
    For longer tasks, one can add to the files containing low-level
    functionality.

4 - Code which describes 'definitions', classes which help users describe
    inputs to pyAFQ. These have particular requirements and must be
    BIDS compatible, see `here <https://yeatmanlab.github.io/pyAFQ/developing/definitions.html>`_ .

5 - Code which contains tests, such as in the tests folder and utils/tests.
    It is important when adding new functionality to add a corresponding test.


.. toctree::
    :maxdepth: 2

    tasks
    definitions
    releasing
