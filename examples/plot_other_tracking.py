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
# dataset (https://purl.stanford.edu/ng782rw8378). The calls below organize the # preprocessed data according fetches
# the results of tractography with this dataset and organizes it within
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
        'tractography_subsampled.trk'),
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
# Now, we can run AFQ, pointing to the derivatives of the
# "my_tractography" pipeline as inputs:

bundle_names = ["SLF", "ARC", "CST", "FP"]


my_afq = api.AFQ(
    bids_path,
    dmriprep='vistasoft',
    bundle_info=bundle_names,
    custom_tractography_bids_filters={
        "suffix": "tractography",
        "scope": "my_tractography"
    })

my_afq.tract_profiles
