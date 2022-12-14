"""
==========================
Free water DTI
==========================

"""
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib

from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd

from AFQ.definitions.mapping import ItkMap
from AFQ.definitions.image import ImageFile

##########################################################################
# Get some example data
#
# XXX We'll need HBN data here.

afd.fetch_hbn_preproc(["NDARAA948VFH"])

brain_mask_definition = ImageFile(
    suffix="mask",
    filters={'desc': 'brain',
             'space': 'T1w',
             'scope': 'qsiprep'})

mapping_definition=ItkMap(
            warp_suffix='xfm',
            warp_filters={'from': 'MNI152NLin2009cAsym',
                          'to': 'T1w',
                          'scope': 'qsiprep'})


myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'HBN'),
    preproc_pipeline='qsiprep',
    mapping_definition = mapping_definition,
    brain_mask_definition = brain_mask_definition,
    scalars=["fwdti_fa", "fwdti_md", "fwdti_fwf", "dti_fa", "dti_md"])

##########################################################################

FWFA_fname = myafq.export("fwdti_fa")["01"]
FWFA_img = nib.load(FA_fname)
FWFA = FWFA_img.get_fdata()

FA_fname = myafq.export("dti_fa")["01"]
FA_img = nib.load(FA_fname)
FA = FA_img.get_fdata()

fig, ax = plt.subplots(1, 2)
ax[0].matshow(FA[:, :, FA.shape[-1] // 2], cmap='gray')
ax[0].axis("off")

ax[0].matshow(FWFA[:, :, FWFA.shape[-1] // 2], cmap='gray')
ax[0].axis("off")
