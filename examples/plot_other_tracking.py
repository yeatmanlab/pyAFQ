"""
=============================================
Segmenting tractography from another pipeline
=============================================

The AFQ API provides facilities to segment tractography results obtained using
other software. For example, we often use
`qsiprep <https://qsiprep.readthedocs.io/en/latest/>`_ to preprocess our data
and reconstruct tractographies with software such as
`MRTRIX <https://www.mrtrix.org/>`_. Here, we will demonstrate how to use
these reconstructions in the pyAFQ segmentation and tractometry pipeline.

"""
import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import plotly

from AFQ import api
import AFQ.data as afd

##########################################################################
# Example data
# ---------------------
# The example data we will use here is generated from the Stanford HARDI
# dataset (https://purl.stanford.edu/ng782rw8378). The calls below fetch
# the results of tractography with this dataset and organize it within
# the `~/AFQ_data` folder.

afd.organize_stanford_data()
afd.fetch_stanford_hardi_tractography()

##########################################################################
# Reorganize data
# ---------------------
# We organize the data so that it conforms with the BIDS standard for
# derivatives:

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


##########################################################################
# Once this is done, you should have a folder in your home directory that
# looks like this:
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


##########################################################################
# We specify the information we need to define the bundles that we will
# segment. In this case, we are going to use a list of bundle names for the
# bundle info. These names refer to bundles for which we already have
# clear definitions of the information needed to segment them (e.g.,
# waypoint ROIs and probability maps). For an example that includes
# custom definition of bundle info, see the `plot_callosal_tract_profile`
# example.

bundle_info = ["SLF", "ARC", "CST", "FP"]

##########################################################################
# Now, we can run AFQ, pointing to the derivatives of the
# "my_tractography" pipeline as inputs. This is done by setting the
# `custom_tractography_bids_filters` key-word argument. We pass the
# `bundle_info` defined above. We also point to the preprocessed
# data that is in a `dmriprep` derivatives folder. These data were
# preprocessed with 'vistasoft', so this is the pipeline we'll point to
# If we were using 'qsiprep', this is where we would pass that
# string instead. If we did that, AFQ would look for a derivatives
# folder called 'stanford_hardi/derivatives/qsiprep' and find the
# preprocessed DWI data within it. Finally, to speed things up
# a bit, we also sub-sample the provided tractography. This is
# done by defining the segmentation_params dictionary input.
# To sub-sample to 10,000 streamlines, we define
# `'nb_streamlines' = 10000`.

my_afq = api.AFQ(
    bids_path,
    dmriprep='vistasoft',
    bundle_info=bundle_info,
    custom_tractography_bids_filters={
        "suffix": "tractography",
        "scope": "my_tractography"
    },
    segmentation_params={'nb_streamlines': 10000})

my_afq.export_all()
