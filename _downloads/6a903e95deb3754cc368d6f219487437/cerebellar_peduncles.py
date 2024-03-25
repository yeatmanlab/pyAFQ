"""
================================
Delineating cerebellar peduncles
================================

The cerebellar peduncles are white matter tracts that connect the cerebellum to
the brainstem and cortex. In this example, we show how to delineate the
cerebellar peduncles using a subject from the Healthy Brain Network dataset.

This how-to will focus on the definition of the Cerebellar Peduncles (CP) based
on [1]_, [2].

"""

import AFQ.data.fetch as afd
import AFQ.api.bundle_dict as abd
from AFQ.api.group import GroupAFQ
from AFQ.definitions.image import RoiImage, ImageFile


"""
We will use a subject from the HBN dataset. When considering the data
for this operation, look to see whether the acquisition volume includes the
cerbellum. If it does not, it will be hard to delineate the CPs.
"""

bids_path = afd.fetch_hbn_afq(["NDARAA948VFH"])[1]

"""
The next line downloads the cerebellar peduncle templates from `Figshare <https://figshare.com/articles/dataset/Regions_of_interest_for_automated_fiber_quantification_of_cerebellar_bundles/23201630>`_.

"""

cp_rois = afd.read_cp_templates()


"""
The following line defines a bundle dictionary for the cerebellar
peduncles. There are three CPs: The ICP, the MCP, and the SCP. Each CP is
defined by two inclusion ROIs and one exclusion ROI. The Inferior CPs are
defined by inclusion ROIs. They do not decussate, so "cross_midline" is set to
False. The Superior CPs are defined by two inclusion ROIs and an exclusion ROI,
where each SCP's most superior inclusion ROI is the other SCP's exclusion ROI.
They decussate, so "cross_midline" is set to True. The Middle CPs are defined
by two inclusion ROIs and they use the SCP intermediate ROIs as exclusion ROIs.
"""

cp_bundles = abd.cerebellar_bd()

"""
The bundle dict has been defined, and now we are ready to run the AFQ pipeline.
In this case, we are using data that has been preprocessed with QSIprep, so
we have a brain mask that was generated from the T1w data of this subject.
"""

brain_mask_definition = ImageFile(
    suffix="mask",
    filters={'desc': 'brain',
             'space': 'T1w',
             'scope': 'qsiprep'})


"""
Next, we define a GroupAFQ object. In this case, the tracking parameters
focus specifically on the CP, by using the ``RoiImage`` class to define the
seed region. We seed extensively in the ROIs that define the CPs.
"""

cp_afq = GroupAFQ(
    name="cp_afq",
    bids_path=bids_path,
    preproc_pipeline="qsiprep",
    brain_mask_definition=brain_mask_definition,
    tracking_params={
        "n_seeds": 4,
        "directions": "prob",
        "odf_model": "CSD",
        "seed_mask": RoiImage()},
    clip_edges=True,
    bundle_info=cp_bundles)


"""
The call to `export("bundles")` triggers the execution of the full pipeline.
"""
cp_afq.export("bundles")

"""

References
----------
.. [1] S. Jossinger, A. Sares, A. Zislis, D. Sury, V. Gracco, M. Ben-Shachar (2022)
       White matter correlates of sensorimotor synchronization in persistent
       developmental stuttering, Journal of Communication Disorders, 95.

.. [2] S. Jossinger, M. Yablonski, O. Amir, M. Ben-Shachar (2023). The
       contributions of the cerebellar peduncles and the frontal aslant tract
       in mediating speech fluency. Neurobiology of Language 2023;
       doi: https://doi.org/10.1162/nol_a_00098

"""
