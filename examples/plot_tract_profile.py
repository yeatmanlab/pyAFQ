"""
==========================
Plotting tract profiles
==========================

An example of tracking and segmenting two tracts, and plotting their tract
profiles for FA (calculated with DTI).

"""
import os.path as op
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import dipy.data as dpd
from dipy.data import fetcher

import AFQ.utils.streamlines as aus
import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti

import AFQ.utils.models as ut
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
import scipy.ndimage as ndim
import AFQ.segmentation as seg

@profile
def patch_up_roi(roi):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : 3D binary array
        The ROI after it has been transformed

    Returns
    -------
    ROI after dilation and hole-filling
    """
    return ndim.binary_fill_holes(ndim.binary_dilation(roi).astype(int))

@profile
def segment(fdata, fbval, fbvec, streamlines, bundles,
            reg_template=None, mapping=None, as_generator=True,
            clip_to_roi=True, **reg_kwargs):
    """
    Segment streamlines into bundles.

    Parameters
    ----------
    fdata, fbval, fbvec : str
        Full path to data, bvals, bvecs

    streamlines : list of 2D arrays
        Each array is a streamline, shape (3, N).

    bundles: dict
        The format is something like::

             {'name': {'ROIs':[img, img], 'rules':[True, True]}}

    reg_template : str or nib.Nifti1Image, optional.
        Template to use for registration (defaults to the MNI T2)

    mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional
        A mapping between DWI space and a template. Defaults to generate this.

    as_generator : bool, optional
        Whether to generate the streamlines here, or return generators.
        Default: True.

    clip_to_roi : bool, optional
        Whether to clip the streamlines between the ROIs
    """
    img, data, gtab, mask = ut.prepare_data(fdata, fbval, fbvec)
    xform_sl = [s for s in dtu.move_streamlines(streamlines,
                                                np.linalg.inv(img.affine))]

    if reg_template is None:
        reg_template = dpd.read_mni_template()

    if mapping is None:
        mapping = reg.syn_register_dwi(fdata, gtab, template=reg_template,
                                       **reg_kwargs)

    if isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
        mapping = reg.read_mapping(mapping, img, reg_template)

    fiber_groups = {}
    for bundle in bundles:
        select_sl = xform_sl
        for ROI, rule in zip(bundles[bundle]['ROIs'],
                             bundles[bundle]['rules']):
            data = ROI.get_data()
            warped_ROI = patch_up_roi(mapping.transform_inverse(
                data,
                interpolation='nearest'))
            # This function requires lists as inputs:
            select_sl = dts.select_by_rois(select_sl,
                                           [warped_ROI.astype(bool)],
                                           [rule])
        # Next, we reorient each streamline according to an ARBITRARY, but
        # CONSISTENT order. To do this, we use the first ROI for which the rule
        # is True as the first one to pass through, and the last ROI for which
        # the rule is True as the last one to pass through:

        # Indices where the 'rule' is True:
        idx = np.where(bundles[bundle]['rules'])

        orient_ROIs = [bundles[bundle]['ROIs'][idx[0][0]],
                       bundles[bundle]['ROIs'][idx[0][-1]]]

        select_sl = dts.orient_by_rois(select_sl,
                                       orient_ROIs[0].get_data(),
                                       orient_ROIs[1].get_data(),
                                       as_generator=True)

        #  XXX Implement clipping to the ROIs
        #  if clip_to_roi:
        #    dts.clip()

        if as_generator:
            fiber_groups[bundle] = select_sl
        else:
            fiber_groups[bundle] = list(select_sl)

    return fiber_groups

@profile
def func():
    dpd.fetch_stanford_hardi()

    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")

    img = nib.load(hardi_fdata)

    print("Calculating DTI...")
    if not op.exists('./dti_FA.nii.gz'):
        dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
                                out_dir='.')
    else:
        dti_params = {'FA': './dti_FA.nii.gz',
                    'params': './dti_params.nii.gz'}

    print("Tracking...")
    if not op.exists('dti_streamlines.trk'):
        streamlines = list(aft.track(dti_params['params']))
        aus.write_trk('./dti_streamlines.trk', streamlines, affine=img.affine)
    else:
        tg = nib.streamlines.load('./dti_streamlines.trk').tractogram
        streamlines = tg.apply_affine(np.linalg.inv(img.affine)).streamlines

    # Use only a small portion of the streamlines, for expedience:
    streamlines = streamlines[::100]

    templates = afd.read_templates()
    bundle_names = ["CST", "ILF"]

    bundles = {}
    for name in bundle_names:
        for hemi in ['_R', '_L']:
            bundles[name + hemi] = {'ROIs': [templates[name + '_roi1' + hemi],
                                            templates[name + '_roi1' + hemi]],
                                    'rules': [True, True]}


    print("Registering to template...")
    MNI_T2_img = dpd.read_mni_template()
    if not op.exists('mapping.nii.gz'):
        import dipy.core.gradients as dpg
        gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
        mapping = reg.syn_register_dwi(hardi_fdata, gtab)
        reg.write_mapping(mapping, './mapping.nii.gz')
    else:
        mapping = reg.read_mapping('./mapping.nii.gz', img, MNI_T2_img)

    print("Segmenting fiber groups...")
    fiber_groups = segment(hardi_fdata,
                            hardi_fbval,
                            hardi_fbvec,
                            streamlines,
                            bundles,
                            reg_template=MNI_T2_img,
                            mapping=mapping,
                            as_generator=False,
                            affine=img.affine)

    FA_img = nib.load(dti_params['FA'])
    FA_data = FA_img.get_data()

    print("Extracting tract profiles...")
    for bundle in bundles:
        fig, ax = plt.subplots(1)
        profile = seg.calculate_tract_profile(FA_data, fiber_groups[bundle])
        ax.plot(profile)
        ax.set_title(bundle)

func()
