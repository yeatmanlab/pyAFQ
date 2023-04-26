"""
====================
How pyAFQ uses BIDS
====================

The pyAFQ API relies heavily on the
`Brain Imaging Data Standard (BIDS) <https://bids-specification.readthedocs.io/en/stable/>`_. This means that the software assumes that its inputs are organized
according to the BIDS spec and its outputs conform where possible with the
BIDS spec.

.. note::

    Derivatives of processing diffusion MRI are not currently fully
    described in the existing BIDS specification, but describing these
    is part of an ongoing effort. Wherever possible, we conform with
    the draft implementation of the BIDS DWI derivatives available
    `here <https://bids-specification.readthedocs.io/en/wip-derivatives/05-derivatives/05-diffusion-derivatives.html>`_

In this example, we will explore the use of BIDS in pyAFQ and see
how BIDS allows us to extend and provide flexibility to the users
of the software.

"""

import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib

from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd


##########################################################################
# To interact with and query BIDS datasets, we use
#  `pyBIDS <https://bids-standard.github.io/pybids/>`_.

import bids


##########################################################################
# We start with some example data. The data we will use here is
# generated from the
# `Stanford HARDI dataset <https://purl.stanford.edu/ng782rw8378>`_.
# The call below fetches
# this dataset and organized it within the `~/AFQ_data` folder in the BIDS
# format.

afd.organize_stanford_data(clear_previous_afq=True)

##########################################################################
# After doing that, we should have a folder that looks like this:
#
# | stanford_hardi
# | ├── dataset_description.json
# | └── derivatives
# |     ├── freesurfer
# |     │   ├── dataset_description.json
# |     │   └── sub-01
# |     │       └── ses-01
# |     │           └── anat
# |     │               ├── sub-01_ses-01_T1w.nii.gz
# |     │               └── sub-01_ses-01_seg.nii.gz
# |     └── vistasoft
# |         ├── dataset_description.json
# |         └── sub-01
# |             └── ses-01
# |                 └── dwi
# |                     ├── sub-01_ses-01_dwi.bvals
# |                     ├── sub-01_ses-01_dwi.bvecs
# |                     └── sub-01_ses-01_dwi.nii.gz
#
# The top level directory is our overall bids dataset folder. In most
# cases, this folder will include a `raw` folder that will contain the
# raw data. In this case, we do not include the raw folder and only have
# the pipelines that contains the outputs of preprocessing the data.
# In general, only the preprocessed diffusion data is required.
# See the "Organizing your data" section of "Using pyAFQ" for more details.
# In this case, one folder containing Freesurfer derivatives and another
# folder containing the DWI data that has been preprocessed with Vistasoft.
# pyAFQ provides facilities to segment tractography results obtained
# using other software. For example, we often use
# `qsiprep <https://qsiprep.readthedocs.io/en/latest/>`_ to preprocess
# our data and reconstruct tractographies with software such as
# `MRTRIX <https://www.mrtrix.org/>`_. Here, we will demonstrate how to use
# these reconstructions in the pyAFQ segmentation and tractometry pipeline
# We fetch this data and add it as a separate pipeline

afd.fetch_stanford_hardi_tractography()

bids_path = op.join(op.expanduser('~'), 'AFQ_data', 'stanford_hardi')
tractography_path = op.join(bids_path, 'derivatives', 'my_tractography')
sub_path = op.join(tractography_path, 'sub-01', 'ses-01', 'dwi')

os.makedirs(sub_path, exist_ok=True)
os.rename(
    op.join(
        op.expanduser('~'),
        'AFQ_data',
        'stanford_hardi_tractography',
        'full_segmented_cleaned_tractography.trk'),
    op.join(
        sub_path,
        'sub-01_ses-01-dwi_tractography.trk'))

afd.to_bids_description(
    tractography_path,
    **{"Name": "my_tractography",
        "PipelineDescription": {"Name": "my_tractography"}})


###########################################################################
# After we do that, our dataset folder should look like this:
#
# | stanford_hardi
# | ├── dataset_description.json
# | └── derivatives
# |     ├── freesurfer
# |     │   ├── dataset_description.json
# |     │   └── sub-01
# |     │       └── ses-01
# |     │           └── anat
# |     │               ├── sub-01_ses-01_T1w.nii.gz
# |     │               └── sub-01_ses-01_seg.nii.gz
# |     ├── my_tractography
# |     |   ├── dataset_description.json
# |     │   └── sub-01
# |     │       └── ses-01
# |     │           └── dwi
# |     │               └── sub-01_ses-01-dwi_tractography.trk
# |     └── vistasoft
# |         ├── dataset_description.json
# |         └── sub-01
# |             └── ses-01
# |                 └── dwi
# |                     ├── sub-01_ses-01_dwi.bvals
# |                     ├── sub-01_ses-01_dwi.bvecs
# |                     └── sub-01_ses-01_dwi.nii.gz
#
# To explore the layout of these derivatives, we will initialize a
# :class:`BIDSLayout` class instance to help us see what is in this dataset

layout = bids.BIDSLayout(bids_path, derivatives=True)

##########################################################################
# Because there is no raw data in this BIDS layout (only derivatives),
# pybids will report that there are no subjects and sessions:

print(layout)

##########################################################################
# But a query on the derivatives will reveal the different derivatives that
# are stored here:

print(layout.derivatives)

##########################################################################
# We can use a :class:`bids.BIDSValidator` object to make sure that the
# files within our data set are BIDS-compliant. For example, we can
# extract the tractography derivatives part of our layout using:

my_tractography = layout.derivatives["my_tractography"]

##########################################################################
# This variable is also a BIDS layout object. This object has a ``get``
# method, which allows us to query and find specific items within the
# layout. For example, we can ask for files that have a suffix consistent
# with tractography results:

tractography_files = my_tractography.get(suffix='tractography')

##########################################################################
# Or ask for files that have a ``.trk`` extension:

tractography_files = my_tractography.get(extension='.trk')

##########################################################################
# In this case, both of these would produce the same result.

tractography_file = tractography_files[0]
print(tractography_file)

##########################################################################
# We can also get some more structured information about this file:

print(tractography_file.get_entities())


##########################################################################
# We can use a :class:`bids.BIDSValidator` class instance to validate that
# this file is compliant with the specification. Note that the validator
# requires that the filename be provided relative to the root of the BIDS
# dataset, so we have to split the string that contains the full path
# of the tractography to extract only the part that is relative to the
# root of the entire BIDS ``layout`` object:

tractography_full_path = tractography_file.path
tractography_relative_path = tractography_full_path.split(layout.root)[-1]

validator = bids.BIDSValidator()
print(validator.is_bids(tractography_relative_path))

##########################################################################
# Next, we specify the information we need to define the bundles that we are
# interested in segmenting. In this case, we are going to use a list of
# bundle names for the bundle info. These names refer to bundles for
# which we already have clear definitions of the information
# needed to segment them (e.g., waypoint ROIs and probability maps).
# For an example that includes custom definition of bundle info, see the
# `plot_callosal_tract_profile example <http://yeatmanlab.github.io/pyAFQ/auto_examples/plot_callosal_tract_profile.html>`_.

bundle_info = [
    "SLF_L", "SLF_R", "ARC_L", "ARC_R",
    "CST_L", "CST_R", "FP"]

##########################################################################
# Now, we can define our GroupAFQ object, pointing to the derivatives of the
# `'my_tractography'` pipeline as inputs. This is done by setting the
# `import_tract` key-word argument. We pass the
# `bundle_info` defined above. We also point to the preprocessed
# data that is in a `'dmriprep'` pipeline. Note that the pipeline name
# is not necessarily the name of the folder it is in; the pipeline name is
# defined in each pipeline's `dataset_description.json`. These data were
# preprocessed with 'vistasoft', so this is the pipeline we'll point to
# If we were using `'qsiprep'`, this is where we would pass that
# string instead. If we did that, AFQ would look for a derivatives
# folder called `'stanford_hardi/derivatives/qsiprep'` and find the
# preprocessed DWI data within it. Finally, to speed things up
# a bit, we also sub-sample the provided tractography. This is
# done by defining the segmentation_params dictionary input.
# To sub-sample to 10,000 streamlines, we define
# `'nb_streamlines' = 10000`.

my_afq = GroupAFQ(
    bids_path,
    preproc_pipeline='vistasoft',
    bundle_info=bundle_info,
    import_tract={
        "suffix": "tractography",
        "scope": "my_tractography"
    },
    segmentation_params={'nb_streamlines': 10000})

##########################################################################
# Finally, to run the segmentation and extract tract profiles, we call
# The `export_all` method. This creates all of the derivative outputs of
# AFQ within the 'stanford_hardi/derivatives/afq' folder.

my_afq.export_all()


##########################################################################
# A few common issues that can hinder BIDS from working properly are:
#
# 1. Faulty `dataset_description.json` file. You need to make sure that the
#    file contains the right names for the pipeline. See above for an example
#    of that.
# 2. File naming convention doesn't uniquely identify file with bids filters.
