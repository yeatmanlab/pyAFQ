# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import pandas as pd
import dask.dataframe as ddf
import glob
import os
import os.path as op
from pathlib import PurePath

import numpy as np
import nibabel as nib

import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu
import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.data as afd
from AFQ.dti import _fit as dti_fit
from AFQ.csd import _fit as csd_fit
import AFQ.tractography as aft
import dipy.reconst.dti as dpy_dti
import AFQ.utils.streamlines as aus
import AFQ.segmentation as seg
import AFQ.registration as reg
import AFQ.utils.volume as auv


__all__ = ["AFQ", "make_bundle_dict"]


def do_preprocessing():
    raise NotImplementedError


BUNDLES = ["ATR", "CGC", "CST", "HCC", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]


# Monkey-patch this in, until https://github.com/nipy/dipy/pull/1695 is
# merged:
def _bundle_profile(data, bundle, affine=None, n_points=100,
                   weights=None):
    """
    Calculates a summarized profile of data for a bundle along its length.

    Follows the approach outlined in [Yeatman2012]_.

    Parameters
    ----------
    data : 3D volume
        The statistic to sample with the streamlines.
    bundle : StreamLines class instance
        The collection of streamlines (possibly already resampled into an array
         for each to have the same length) with which we are resampling. See
         Note below about orienting the streamlines.
    weights : 1D
        array or 2D array (optional) Weight each streamline (1D) or each
        node (2D) when calculating the tract-profiles. Must sum to 1 across
        streamlines (in each node if relevant).

    Returns
    -------
    ndarray : a 1D array with the profile of `data` along the length of
        `bundle`

    Note
    ----
    Before providing a bundle as input to this function, you will need to make
    sure that the streamlines in the bundle are all oriented in the same
    orientation relative to the bundle (use :func:`orient_by_streamline`).

    References
    ----------
    .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty, Nathaniel J. Myall,
       Brian A. Wandell, and Heidi M. Feldman. 2012. "Tract Profiles of White
       Matter Properties: Automating Fiber-Tract Quantification" PloS One 7
       (11): e49790.
    """
    if len(bundle) == 0:
        raise ValueError("The bundle contains no streamlines")

    # Resample each streamline to the same number of points:
    fgarray = dts.set_number_of_points(bundle, n_points)

    # Extract the values
    values = np.array(dts.values_from_volume(data, fgarray, affine=affine))

    if weights is None:
        weights = np.ones(values.shape) / values.shape[0]
    else:
        # We check that weights *always sum to 1 across streamlines*:
        if not np.allclose(np.sum(weights, 0), np.ones(n_points)):
            raise ValueError("The sum of weights across streamlines must ",
                             "be equal to 1")

    return np.sum(weights * values, 0)


dts.bundle_profile = _bundle_profile


def make_bundle_dict(bundle_names=BUNDLES, seg_algo="planes"):
    """
    Create a bundle dictionary, needed for the segmentation

    Parameters
    ----------
    bundle_names : list, optional
        A list of the bundles to be used in this case. Default: all of them
    """
    if seg_algo == "planes":
        templates = afd.read_templates()
        callosal_templates = afd.read_callosum_templates()
        # For the arcuate, we need to rename a few of these and duplicate the
        # SLF ROI:
        templates['ARC_roi1_L'] = templates['SLF_roi1_L']
        templates['ARC_roi1_R'] = templates['SLF_roi1_R']
        templates['ARC_roi2_L'] = templates['SLFt_roi2_L']
        templates['ARC_roi2_R'] = templates['SLFt_roi2_R']

        afq_bundles = {}
        # Each bundles gets a digit identifier (to be stored in the tractogram)
        uid = 1
        for name in bundle_names:
            # Consider hard coding since we might have different rules for
            # some tracts
            if name in ["FA", "FP"]:
                afq_bundles[name] = {
                    'ROIs': [templates[name + "_L"],
                             templates[name + "_R"],
                             callosal_templates["Callosum_midsag"]],
                    'rules': [True, True, True],
                    'prob_map': templates[name + "_prob_map"],
                    'cross_midline': True,
                    'uid': uid}
                uid += 1
            # SLF is a special case, because it has an exclusion ROI:
            elif name == "SLF":
                for hemi in ['_R', '_L']:
                    afq_bundles[name + hemi] = {
                        'ROIs': [templates[name + '_roi1' + hemi],
                                 templates[name + '_roi2' + hemi],
                                 templates["SLFt_roi2" + hemi]],
                        'rules': [True, True, False],
                        'prob_map': templates[name + hemi + '_prob_map'],
                        'cross_midline': False,
                        'uid': uid}
                    uid += 1
            else:
                for hemi in ['_R', '_L']:
                    afq_bundles[name + hemi] = {
                        'ROIs': [templates[name + '_roi1' + hemi],
                                 templates[name + '_roi2' + hemi]],
                        'rules': [True, True],
                        'prob_map': templates[name + hemi + '_prob_map'],
                        'cross_midline': False,
                        'uid': uid}

                    uid += 1

    elif seg_algo == "recobundles":
        afq_bundles = {}
        uid = 1
        bundle_dict = afd.read_hcp_atlas_16_bundles()
        afq_bundles["whole_brain"] = bundle_dict["whole_brain"]
        for name in bundle_names:
            if name in ['CCMid', 'CC_ForcepsMajor', 'CC_ForcepsMinor', 'MCP']:
                afq_bundles[name] = bundle_dict[name]
                afq_bundles[name]['uid'] = uid
                uid += 1
            else:
                for hemi in ["_R", "_L"]:
                    afq_bundles[name + hemi] = bundle_dict[name + hemi]
                    afq_bundles[name + hemi]['uid'] = uid
                    uid += 1
    else:
        raise ValueError("Input: %s is not a valid input`seg_algo`" % seg_algo)

    return afq_bundles


def _b0(row, force_recompute=False):
    b0_file = _get_fname(row, '_b0.nii.gz')
    if not op.exists(b0_file) or force_recompute:
        img = nib.load(row['dwi_file'])
        data = img.get_fdata()
        gtab = row['gtab']
        mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
        mean_b0_img = nib.Nifti1Image(mean_b0, img.affine)
        nib.save(mean_b0_img, b0_file)
    return b0_file


def _brain_mask(row, median_radius=4, numpass=1, autocrop=False,
                vol_idx=None, dilate=10, force_recompute=False):
    brain_mask_file = _get_fname(row, '_brain_mask.nii.gz')
    if not op.exists(brain_mask_file) or force_recompute:
        mean_b0_img = nib.load(_b0(row))
        mean_b0 = mean_b0_img.get_fdata()
        _, brain_mask = median_otsu(mean_b0, median_radius, numpass,
                                    autocrop, dilate=dilate)
        be_img = nib.Nifti1Image(brain_mask.astype(int),
                                 mean_b0_img.affine)
        nib.save(be_img, brain_mask_file)
    return brain_mask_file


def _dti_fit(row):
    dti_params_file = _dti(row)
    dti_params = nib.load(dti_params_file).get_fdata()
    tm = dpy_dti.TensorModel(row['gtab'])
    tf = dpy_dti.TensorFit(tm, dti_params)
    return tf


def _dti(row, force_recompute=False):
    dti_params_file = _get_fname(row, '_dti_params.nii.gz')
    if not op.exists(dti_params_file) or force_recompute:
        img = nib.load(row['dwi_file'])
        data = img.get_fdata()
        gtab = row['gtab']
        brain_mask_file = _brain_mask(row)
        mask = nib.load(brain_mask_file).get_fdata()
        dtf = dti_fit(gtab, data, mask=mask)
        nib.save(nib.Nifti1Image(dtf.model_params, row['dwi_affine']),
                 dti_params_file)
    return dti_params_file


def _csd(row, force_recompute=False, response=None,
         sh_order=8, lambda_=1, tau=0.1,):
    csd_params_file = _get_fname(row, '_csd_params.nii.gz')
    if not op.exists(csd_params_file) or force_recompute:
        img = nib.load(row['dwi_file'])
        data = img.get_fdata()
        gtab = row['gtab']
        brain_mask_file = _brain_mask(row)
        mask = nib.load(brain_mask_file).get_fdata()
        csdf = csd_fit(gtab, data, mask=mask,
                       response=response, sh_order=sh_order,
                       lambda_=lambda_, tau=tau)
        nib.save(nib.Nifti1Image(csdf.shm_coeff, row['dwi_affine']),
                 csd_params_file)
    return csd_params_file


def _dti_fa(row, force_recompute=False):
    dti_fa_file = _get_fname(row, '_dti_fa.nii.gz')
    if not op.exists(dti_fa_file) or force_recompute:
        tf = _dti_fit(row)
        fa = tf.fa
        nib.save(nib.Nifti1Image(fa, row['dwi_affine']),
                 dti_fa_file)
    return dti_fa_file


def _dti_cfa(row, force_recompute=False):
    dti_cfa_file = _get_fname(row, '_dti_cfa.nii.gz')
    if not op.exists(dti_cfa_file) or force_recompute:
        tf = _dti_fit(row)
        cfa = tf.color_fa
        nib.save(nib.Nifti1Image(cfa, row['dwi_affine']),
                 dti_cfa_file)
    return dti_cfa_file


def _dti_pdd(row, force_recompute=False):
    dti_pdd_file = _get_fname(row, '_dti_pdd.nii.gz')
    if not op.exists(dti_pdd_file) or force_recompute:
        tf = _dti_fit(row)
        pdd = tf.directions.squeeze()
        # Invert the x coordinates:
        pdd[..., 0] = pdd[..., 0] * -1

        nib.save(nib.Nifti1Image(pdd, row['dwi_affine']),
                 dti_pdd_file)
    return dti_pdd_file


def _dti_md(row, force_recompute=False):
    dti_md_file = _get_fname(row, '_dti_md.nii.gz')
    if not op.exists(dti_md_file) or force_recompute:
        tf = _dti_fit(row)
        md = tf.md
        nib.save(nib.Nifti1Image(md, row['dwi_affine']),
                 dti_md_file)
    return dti_md_file


# Keep track of functions that compute scalars:
_scalar_dict = {"dti_fa": _dti_fa,
                "dti_md": _dti_md}


def _reg_prealign(row, force_recompute=False):
    prealign_file = _get_fname(row, '_reg_prealign.npy')
    if not op.exists(prealign_file) or force_recompute:
        moving = nib.load(_b0(row, force_recompute=force_recompute))
        static = dpd.read_mni_template()
        moving_data = moving.get_fdata()
        moving_affine = moving.affine
        static_data = static.get_fdata()
        static_affine = static.affine
        _, aff = reg.affine_registration(moving_data,
                                         static_data,
                                         moving_affine,
                                         static_affine)
        np.save(prealign_file, aff)
    return prealign_file


def _mapping(row, reg_template, force_recompute=False):
    mapping_file = _get_fname(row, '_mapping.nii.gz')
    if not op.exists(mapping_file) or force_recompute:
        gtab = row['gtab']
        reg_prealign = np.load(_reg_prealign(
            row,
            force_recompute=force_recompute))
        warped_b0, mapping = reg.syn_register_dwi(row['dwi_file'], gtab,
                                                  template=reg_template,
                                                  prealign=reg_prealign)
        mapping.codomain_world2grid = np.linalg.inv(reg_prealign)
        reg.write_mapping(mapping, mapping_file)
    return mapping_file


def _streamlines(row, wm_labels, odf_model="DTI", directions="det",
                 n_seeds=2, random_seeds=False, force_recompute=False,
                 wm_fa_thresh=0.2):
    """
    wm_labels : list
        The values within the segmentation that are considered white matter. We
        will use this part of the image both to seed tracking (seeding
        throughout), and for stopping.
    """
    streamlines_file = _get_fname(row,
                                  '%s_%s_streamlines.trk' % (odf_model,
                                                             directions))
    if not op.exists(streamlines_file) or force_recompute:
        if odf_model == "DTI":
            params_file = _dti(row)
        elif odf_model == "CSD":
            params_file = _csd(row)

        dwi_img = nib.load(row['dwi_file'])
        dwi_data = dwi_img.get_fdata()

        if 'seg_file' in row.index:
            # If we found a white matter segmentation in the
            # expected location:
            seg_img = nib.load(row['seg_file'])
            seg_data_orig = seg_img.get_fdata()
            # For different sets of labels, extract all the voxels that
            # have any of these values:
            wm_mask = np.sum(np.concatenate([(seg_data_orig == l)[..., None]
                                            for l in wm_labels], -1), -1)

            # Resample to DWI data:
            wm_mask = np.round(reg.resample(wm_mask, dwi_data[..., 0],
                                            seg_img.affine,
                                            dwi_img.affine)).astype(int)
        else:
            # Otherwise, we'll identify the white matter based on FA:
            dti_fa = nib.load(_dti_fa(row)).get_fdata()
            wm_mask = dti_fa > wm_fa_thresh

        streamlines = aft.track(params_file,
                                directions=directions,
                                n_seeds=n_seeds,
                                random_seeds=random_seeds,
                                seed_mask=wm_mask,
                                stop_mask=wm_mask)

        save_tractogram(streamlines_file,
                        dtu.move_streamlines(streamlines,
                                             np.linalg.inv(dwi_img.affine)),
                        dwi_img.affine)

    return streamlines_file


def _recobundles(row, wm_labels, bundle_dict, reg_template, odf_model="DTI",
                 directions="det", n_seeds=2, random_seeds=False,
                 force_recompute=False):

    bundles_file = _get_fname(row,
                              '%s_%s_bundles.trk' % (odf_model,
                                                     directions))
    if not op.exists(bundles_file) or force_recompute:
        streamlines_file = _streamlines(row, wm_labels,
                                        odf_model=odf_model,
                                        directions=directions,
                                        n_seeds=n_seeds,
                                        random_seeds=random_seeds,
                                        force_recompute=force_recompute)
        tg = nib.streamlines.load(streamlines_file).tractogram
        sl = tg.apply_affine(np.linalg.inv(row['dwi_affine'])).streamlines
        segmentation = seg.Segmentation(method='Reco')
        bundles = segmentation.segment(bundle_dict, sl)
        tgram = aus.bundles_to_tgram(bundles, bundle_dict, row['dwi_affine'])
        nib.streamlines.save(tgram, bundles_file)
    return bundles_file


def _bundles(row, wm_labels, bundle_dict, reg_template, odf_model="DTI",
             directions="det", n_seeds=2, random_seeds=False,
             force_recompute=False):
    bundles_file = _get_fname(row,
                              '%s_%s_bundles.trk' % (odf_model,
                                                     directions))
    if not op.exists(bundles_file) or force_recompute:
        streamlines_file = _streamlines(row, wm_labels,
                                        odf_model=odf_model,
                                        directions=directions,
                                        n_seeds=n_seeds,
                                        random_seeds=random_seeds,
                                        force_recompute=force_recompute)
        tg = nib.streamlines.load(streamlines_file).tractogram
        sl = tg.apply_affine(np.linalg.inv(row['dwi_affine'])).streamlines

        reg_prealign = np.load(_reg_prealign(row,
                                             force_recompute=force_recompute))

        mapping = reg.read_mapping(_mapping(row, reg_template),
                                   row['dwi_file'],
                                   reg_template,
                                   prealign=np.linalg.inv(reg_prealign))

        segmentation = seg.Segmentation()
        bundles = segmentation.segment(row['dwi_file'],
                              row['bval_file'],
                              row['bvec_file'],
                              bundle_dict,
                              sl,
                              reg_template=reg_template,
                              mapping=mapping)

        tgram = aus.bundles_to_tgram(bundles, bundle_dict, row['dwi_affine'])
        nib.streamlines.save(tgram, bundles_file)
    return bundles_file


def _clean_bundles(row, wm_labels, bundle_dict, reg_template, odf_model="DTI",
                   directions="det", n_seeds=2, random_seeds=False,
                   force_recompute=False):
    clean_bundles_file = _get_fname(row,
                                    '%s_%s_clean_bundles.trk' % (odf_model,
                                                                 directions))
    if not op.exists(clean_bundles_file) or force_recompute:
        bundles_file = _bundles(row,
                                wm_labels,
                                bundle_dict,
                                reg_template,
                                odf_model=odf_model,
                                directions=directions,
                                n_seeds=n_seeds,
                                random_seeds=random_seeds,
                                force_recompute=False)
        tg = nib.streamlines.load(bundles_file).tractogram
        sl = tg.apply_affine(np.linalg.inv(row['dwi_affine'])).streamlines
        tgram = nib.streamlines.Tractogram([], {'bundle': []})
        for b in bundle_dict.keys():
            idx = np.where(tg.data_per_streamline['bundle']
                           == bundle_dict[b]['uid'])[0]
            this_sl = sl[idx]
            this_sl = seg.clean_fiber_group(this_sl)
            this_tgram = nib.streamlines.Tractogram(
                this_sl,
                data_per_streamline={
                    'bundle': (len(this_sl)
                               * [bundle_dict[b]['uid']])},
                    affine_to_rasmm=row['dwi_affine'])
            tgram = aus.add_bundles(tgram, this_tgram)
        nib.streamlines.save(tgram, clean_bundles_file)

    return clean_bundles_file


def _tract_profiles(row, wm_labels, bundle_dict, reg_template,
                    odf_model="DTI", directions="det",
                    n_seeds=2, random_seeds=False,
                    scalars=["dti_fa", "dti_md"], weighting=None,
                    force_recompute=False):
    profiles_file = _get_fname(row, '_profiles.csv')
    if not op.exists(profiles_file) or force_recompute:
        bundles_file = _clean_bundles(row,
                                      wm_labels,
                                      bundle_dict,
                                      reg_template,
                                      odf_model=odf_model,
                                      directions=directions,
                                      n_seeds=n_seeds,
                                      random_seeds=random_seeds,
                                      force_recompute=force_recompute)
        keys = []
        vals = []
        for k in bundle_dict.keys():
            keys.append(bundle_dict[k]['uid'])
            vals.append(k)
        reverse_dict = dict(zip(keys, vals))

        bundle_names = []
        profiles = []
        node_numbers = []
        scalar_names = []

        trk = nib.streamlines.load(bundles_file)
        for scalar in scalars:
            scalar_file = _scalar_dict[scalar](row,
                                               force_recompute=force_recompute)
            scalar_data = nib.load(scalar_file).get_fdata()
            for b in np.unique(trk.tractogram.data_per_streamline['bundle']):
                idx = np.where(
                    trk.tractogram.data_per_streamline['bundle'] == b)[0]
                this_sl = list(trk.streamlines[idx])
                bundle_name = reverse_dict[b]
                this_profile = dts.bundle_profile(
                    scalar_data,
                    this_sl,
                    affine=row['dwi_affine'])
                nodes = list(np.arange(this_profile.shape[0]))
                bundle_names.extend([bundle_name] * len(nodes))
                node_numbers.extend(nodes)
                scalar_names.extend([scalar] * len(nodes))
                profiles.extend(list(this_profile))

        profile_dframe = pd.DataFrame(dict(profiles=profiles,
                                           bundle=bundle_names,
                                           node=node_numbers,
                                           scalar=scalar_names))
        profile_dframe.to_csv(profiles_file)

    return profiles_file


def _template_xform(row, reg_template, force_recompute=False):
    template_xform_file = _get_fname(row, "_template_xform.nii.gz")
    if not op.exists(template_xform_file) or force_recompute:
        reg_prealign = np.load(_reg_prealign(row,
                               force_recompute=force_recompute))
        mapping = reg.read_mapping(_mapping(row,
                                            reg_template,
                                            force_recompute=force_recompute),
                                   row['dwi_file'],
                                   reg_template,
                                   prealign=np.linalg.inv(reg_prealign))

        template_xform = mapping.transform_inverse(reg_template.get_fdata())
        nib.save(nib.Nifti1Image(template_xform,
                                 row['dwi_affine']),
                 template_xform_file)
    return template_xform_file


def _export_rois(row, bundle_dict, reg_template, force_recompute=False):
    reg_prealign = np.load(_reg_prealign(row,
                                         force_recompute=force_recompute))

    mapping = reg.read_mapping(_mapping(row, reg_template),
                               row['dwi_file'],
                               reg_template,
                               prealign=np.linalg.inv(reg_prealign))

    rois_dir = op.join(row['results_dir'], 'ROIs')
    os.makedirs(rois_dir, exist_ok=True)

    for bundle in bundle_dict:
        for ii, roi in enumerate(bundle_dict[bundle]['ROIs']):

            if bundle_dict[bundle]['rules'][ii]:
                inclusion = 'include'
            else:
                inclusion = 'exclude'
            fname = op.join(
                rois_dir, '%s_roi%s_%s.nii.gz' % (bundle, ii + 1, inclusion))
            warped_roi = auv.patch_up_roi(
                (mapping.transform_inverse(
                    roi.get_fdata(),
                    interpolation='linear')) > 0).astype(int)
            # Cast to float32, so that it can be read in by MI-Brain:
            nib.save(nib.Nifti1Image(warped_roi.astype(np.float32),
                                     row['dwi_affine']),
                     fname)


def _export_bundles(row, wm_labels, bundle_dict, reg_template,
                    odf_model="DTI", directions="det", n_seeds=2,
                    random_seeds=False, force_recompute=False):

    for func, folder in zip([_clean_bundles, _bundles],
                            ['clean_bundles', 'bundles']):
        bundles_file = func(row,
                            wm_labels,
                            bundle_dict,
                            reg_template,
                            odf_model=odf_model,
                            directions=directions,
                            n_seeds=n_seeds,
                            random_seeds=random_seeds,
                            force_recompute=force_recompute)

        bundles_dir = op.join(row['results_dir'], folder)
        os.makedirs(bundles_dir, exist_ok=True)
        trk = nib.streamlines.load(bundles_file)
        tg = trk.tractogram
        streamlines = tg.streamlines
        for bundle in bundle_dict:
            uid = bundle_dict[bundle]['uid']
            idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
            this_sl = dtu.transform_tracking_output(
                streamlines[idx],
                np.linalg.inv(row['dwi_affine']))

            this_tgm = StatefulTractogram(this_sl, row['dwi_img'], Space.VOX)
            fname = op.join(bundles_dir, '%s.trk' % bundle)
            save_tractogram(this_tgm, fname, bbox_valid_check=False)


def _get_affine(fname):
    return nib.load(fname).affine


def _get_fname(row, suffix):
    split_fdwi = op.split(row['dwi_file'])
    fname = op.join(row['results_dir'], split_fdwi[1].split('.')[0]
                    + suffix)
    return fname


class AFQ(object):
    """

    This is file folder structure that AFQ requires in your study folder::

|    study
|      ├-derivatives
|            ├-dmriprep
|                ├── sub01
|                │   ├── sess01
|                │   │   ├── anat
|                │   │   │   ├── sub-01_sess-01_aparc+aseg.nii.gz
|                │   │   │   └── sub-01_sess-01_T1.nii.gz
|                │   │   └── dwi
|                │   │       ├── sub-01_sess-01_dwi.bvals
|                │   │       ├── sub-01_sess-01_dwi.bvecs
|                │   │       └── sub-01_sess-01_dwi.nii.gz
|                │   └── sess02
|                │       ├── anat
|                │       │   ├── sub-01_sess-02_aparc+aseg.nii.gz
|                │       │   └── sub-01_sess-02_T1w.nii.gz
|                │       └── dwi
|                │           ├── sub-01_sess-02_dwi.bvals
|                │           ├── sub-01_sess-02_dwi.bvecs
|                │           └── sub-01_sess-02_dwi.nii.gz
|                └── sub02
|                   ├── sess01
|                   │   ├── anat
|                   │       ├── sub-02_sess-01_aparc+aseg.nii.gz
|                   │   │   └── sub-02_sess-01_T1w.nii.gz
|                   │   └── dwi
|                   │       ├── sub-02_sess-01_dwi.bvals
|                   │       ├── sub-02_sess-01_dwi.bvecs
|                   │       └── sub-02_sess-01_dwi.nii.gz
|                   └── sess02
|                       ├── anat
|                       │   ├── sub-02_sess-02_aparc+aseg.nii.gz
|                       │   └── sub-02_sess-02_T1w.nii.gz
|                       └── dwi
|                           ├── sub-02_sess-02_dwi.bvals
|                           ├── sub-02_sess-02_dwi.bvecs
|                           └── sub-02_sess-02_dwi.nii.gz

    This structure can be automatically generated from BIDS-compliant
    data [1]_, using the dmriprep software [2]_ and BIDS app.

    Notes
    -----
    The structure of the file-system required here resembles that specified
    by BIDS [1]_. In the future, this will be organized according to the
    BIDS derivatives specification, as we require preprocessed, rather than
    raw data.

    .. [1] Gorgolewski et al. (2016). The brain imaging data structure,
           a format for organizing and describing outputs of neuroimaging
           experiments. Scientific Data, 3::160044. DOI: 10.1038/sdata.2016.44.

    .. [2] https://github.com/nipy/dmriprep

    """
    def __init__(self,
                 dmriprep_path,
                 seg_algo="planes",
                 sub_prefix="sub",
                 dwi_folder="dwi",
                 dwi_file="*dwi",
                 anat_folder="anat",
                 anat_file="*T1w*",
                 seg_file='*aparc+aseg*',
                 b0_threshold=0,
                 odf_model="CSD",
                 directions="det",
                 n_seeds=2,
                 random_seeds=False,
                 bundle_names=BUNDLES,
                 dask_it=False,
                 force_recompute=False,
                 reg_template=None,
                 wm_labels=[250, 251, 252, 253, 254, 255, 41, 2, 16, 77]):
        """

        dmriprep_path: str
            The path to the preprocessed diffusion data.

        seg_algo : str
            Which algorithm to use for segmentation.
            Can be one of: {"planes", "recobundles"}

        b0_threshold : int, optional
            The value of b under which it is considered to be b0. Default: 0.

        odf_model : string, optional
            Which model to use for determining directions in tractography
            {"DTI", "DKI", "CSD"}. Default: "DTI"

        directions : string, optional
            How to select directions for tracking (deterministic or
            probablistic) {"det", "prob"}. Default: "det".

        dask_it : bool, optional
            Whether to use a dask DataFrame object

        force_recompute : bool, optional
            Whether to ignore previous results, and recompute all, or not.

        wm_labels : list, optional
            A list of the labels of the white matter in the segmentation file
            used. Default: the white matter values for the segmentation
            provided with the HCP data, including labels for midbraing:
            [250, 251, 252, 253, 254, 255, 41, 2, 16, 77].
        """
        self.directions = directions
        self.odf_model = odf_model
        self.bundle_dict = make_bundle_dict(bundle_names=bundle_names,
                                            seg_algo=seg_algo)
        self.seg_algo = seg_algo
        self.force_recompute = force_recompute
        self.wm_labels = wm_labels
        self.n_seeds = n_seeds
        self.random_seeds = random_seeds
        if reg_template is None:
            self.reg_template = dpd.read_mni_template()
        else:
            if not isinstance(reg_template, nib.Nifti1Image):
                reg_template = nib.load(reg_template)
            self.reg_template = reg_template
        # This is the place in which each subject's full data lives
        self.dmriprep_dirs = glob.glob(op.join(dmriprep_path,
                                               '%s*' % sub_prefix))

        # This is where all the outputs will go:
        self.afq_dir = op.join(
            op.join(*PurePath(dmriprep_path).parts[:-1]), 'afq')

        os.makedirs(self.afq_dir, exist_ok=True)

        self.subjects = [op.split(p)[-1] for p in self.dmriprep_dirs]

        sub_list = []
        sess_list = []
        dwi_file_list = []
        bvec_file_list = []
        bval_file_list = []
        anat_file_list = []
        seg_file_list = []
        results_dir_list = []
        for subject, sub_dir in zip(self.subjects, self.dmriprep_dirs):
            sessions = glob.glob(op.join(sub_dir, '*'))
            for sess in sessions:
                results_dir_list.append(op.join(self.afq_dir,
                                        subject,
                                        PurePath(sess).parts[-1]))

                os.makedirs(results_dir_list[-1], exist_ok=True)

                dwi_file_list.append(glob.glob(op.join(sub_dir,
                                                       ('%s/%s/%s.nii.gz' %
                                                        (sess, dwi_folder,
                                                         dwi_file))))[0])

                bvec_file_list.append(glob.glob(op.join(sub_dir,
                                                        ('%s/%s/%s.bvec*' %
                                                         (sess, dwi_folder,
                                                          dwi_file))))[0])

                bval_file_list.append(glob.glob(op.join(sub_dir,
                                                        ('%s/%s/%s.bval*' %
                                                         (sess, dwi_folder,
                                                          dwi_file))))[0])

                # The following two may or may not exist:
                this_anat_file = glob.glob(op.join(sub_dir,
                                           ('%s/%s/%s.nii.gz' %
                                            (sess,
                                             anat_folder,
                                             anat_file))))
                if len(this_anat_file):
                    anat_file_list.append(this_anat_file[0])

                this_seg_file = glob.glob(op.join(sub_dir,
                                                  ('%s/%s/%s.nii.gz' %
                                                   (sess,
                                                    anat_folder,
                                                    seg_file))))
                if len(this_seg_file):
                    seg_file_list.append(this_seg_file[0])

                sub_list.append(subject)
                sess_list.append(sess)
        self.data_frame = pd.DataFrame(dict(subject=sub_list,
                                            dwi_file=dwi_file_list,
                                            bvec_file=bvec_file_list,
                                            bval_file=bval_file_list,
                                            sess=sess_list,
                                            results_dir=results_dir_list))
        # Add these if they exist:
        if len(seg_file_list):
            self.data_frame['seg_file'] = seg_file_list
        if len(anat_file_list):
            self.data_frame['anat_file'] = anat_file_list

        if dask_it:
            self.data_frame = ddf.from_pandas(self.data_frame,
                                              npartitions=len(sub_list))
        self.set_gtab(b0_threshold)
        self.set_dwi_affine()
        self.set_dwi_img()

    def set_gtab(self, b0_threshold):
        self.data_frame['gtab'] = self.data_frame.apply(
            lambda x: dpg.gradient_table(x['bval_file'], x['bvec_file'],
                                         b0_threshold=b0_threshold),
            axis=1)

    def get_gtab(self):
        return self.data_frame['gtab']

    gtab = property(get_gtab, set_gtab)

    def set_dwi_affine(self):
        self.data_frame['dwi_affine'] =\
            self.data_frame['dwi_file'].apply(_get_affine)

    def get_dwi_affine(self):
        return self.data_frame['dwi_affine']

    dwi_affine = property(get_dwi_affine, set_dwi_affine)

    def set_dwi_img(self):
        self.data_frame['dwi_img'] =\
            self.data_frame['dwi_file'].apply(nib.load)

    def get_dwi_img(self):
        return self.data_frame['dwi_img']

    dwi_img = property(get_dwi_img, set_dwi_img)

    def __getitem__(self, k):
        return self.data_frame.__getitem__(k)

    def set_b0(self):
        if ('b0_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['b0_file'] =\
                self.data_frame.apply(_b0,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_b0(self):
        self.set_b0()
        return self.data_frame['b0_file']

    b0 = property(get_b0, set_b0)

    def set_brain_mask(self, median_radius=4, numpass=4, autocrop=False,
                       vol_idx=None, dilate=None):
        if ('brain_mask_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(_brain_mask,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_brain_mask(self):
        self.set_brain_mask()
        return self.data_frame['brain_mask_file']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_dti(self):
        if ('dti_params_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['dti_params_file'] =\
                self.data_frame.apply(_dti,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti_params_file']

    dti = property(get_dti, set_dti)

    def set_dti_fa(self):
        if ('dti_fa_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['dti_fa_file'] =\
                self.data_frame.apply(_dti_fa,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_fa(self):
        self.set_dti_fa()
        return self.data_frame['dti_fa_file']

    dti_fa = property(get_dti_fa, set_dti_fa)

    def set_dti_cfa(self):
        if ('dti_cfa_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['dti_cfa_file'] =\
                self.data_frame.apply(_dti_cfa,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_cfa(self):
        self.set_dti_cfa()
        return self.data_frame['dti_cfa_file']

    dti_cfa = property(get_dti_cfa, set_dti_cfa)

    def set_dti_pdd(self):
        if ('dti_pdd_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['dti_pdd_file'] =\
                self.data_frame.apply(_dti_pdd,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_pdd(self):
        self.set_dti_pdd()
        return self.data_frame['dti_pdd_file']

    dti_pdd = property(get_dti_pdd, set_dti_pdd)

    def set_dti_md(self):
        if ('dti_md_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['dti_md_file'] =\
                self.data_frame.apply(_dti_md,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_md(self):
        self.set_dti_md()
        return self.data_frame['dti_md_file']

    dti_md = property(get_dti_md, set_dti_md)

    def set_mapping(self):
        if 'mapping' not in self.data_frame.columns or self.force_recompute:
            self.data_frame['mapping'] =\
                self.data_frame.apply(_mapping,
                                      args=[self.reg_template],
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_mapping(self):
        self.set_mapping()
        return self.data_frame['mapping']

    mapping = property(get_mapping, set_mapping)

    def set_streamlines(self):
        if ('streamlines_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['streamlines_file'] =\
                self.data_frame.apply(_streamlines, axis=1,
                                      args=[self.wm_labels],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      n_seeds=self.n_seeds,
                                      random_seeds=self.random_seeds,
                                      force_recompute=self.force_recompute)

    def get_streamlines(self):
        self.set_streamlines()
        return self.data_frame['streamlines_file']

    streamlines = property(get_streamlines, set_streamlines)

    def set_bundles(self):
        if self.seg_algo == "planes":
            seg_function = _bundles
        elif self.seg_algo == "recobundles":
            seg_function = _recobundles
        column_exists = 'bundles_file' in self.data_frame.columns
        if (not column_exists or self.force_recompute):
            self.data_frame['bundles_file'] =\
                self.data_frame.apply(seg_function, axis=1,
                                      args=[self.wm_labels,
                                            self.bundle_dict,
                                            self.reg_template],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      n_seeds=self.n_seeds,
                                      random_seeds=self.random_seeds,
                                      force_recompute=self.force_recompute)

    def get_bundles(self):
        self.set_bundles()
        return self.data_frame['bundles_file']

    bundles = property(get_bundles, set_bundles)

    def set_clean_bundles(self):
        column_exists = 'clean_bundles_file' in self.data_frame.columns
        if (not column_exists or self.force_recompute):
            if self.seg_algo == "recobundles":
                self.data_frame['clean_bundles_file'] =\
                    self.data_frame['bundles_file']
            else:
                self.data_frame['clean_bundles_file'] =\
                    self.data_frame.apply(_clean_bundles, axis=1,
                                          args=[self.wm_labels,
                                                self.bundle_dict,
                                                self.reg_template],
                                          odf_model=self.odf_model,
                                          directions=self.directions,
                                          n_seeds=self.n_seeds,
                                          random_seeds=self.random_seeds,
                                          force_recompute=self.force_recompute)

    def get_clean_bundles(self):
        self.set_clean_bundles()
        return self.data_frame['clean_bundles_file']

    clean_bundles = property(get_clean_bundles, set_clean_bundles)

    def set_tract_profiles(self):
        if ('tract_profiles_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['tract_profiles_file'] =\
                self.data_frame.apply(_tract_profiles,
                                      args=[self.wm_labels,
                                            self.bundle_dict,
                                            self.reg_template],
                                      force_recompute=self.force_recompute,
                                      axis=1)

    def get_tract_profiles(self):
        self.set_tract_profiles()
        return self.data_frame['tract_profiles_file']

    tract_profiles = property(get_tract_profiles, set_tract_profiles)

    def set_template_xform(self):
        if ('template_xform_file' not in self.data_frame.columns
                or self.force_recompute):
            self.data_frame['template_xform_file'] = \
                self.data_frame.apply(_template_xform,
                                      args=[self.reg_template],
                                      force_recompute=self.force_recompute,
                                      axis=1)

    def get_template_xform(self):
        self.set_template_xform()
        return self.data_frame['template_xform_file']

    template_xform = property(get_template_xform, set_template_xform)

    def export_rois(self):
        self.data_frame.apply(_export_rois,
                              args=[self.bundle_dict,
                                    self.reg_template],
                              axis=1)

    def export_bundles(self):
        self.data_frame.apply(_export_bundles,
                              args=[self.wm_labels,
                                    self.bundle_dict,
                                    self.reg_template,
                                    self.odf_model,
                                    self.directions,
                                    self.n_seeds,
                                    self.random_seeds,
                                    self.force_recompute],
                              axis=1)

    def combine_profiles(self):
        dfs = []
        for ii, fname in enumerate(self.tract_profiles):
            profiles = pd.read_csv(fname)
            profiles['sub'] = self.data_frame['subject'].iloc[ii]
            profiles['sess'] = op.split(self.data_frame['sess'].iloc[ii])[-1]
            dfs.append(profiles)

        df = pd.concat(dfs)
        df.to_csv(op.join(self.afq_dir, 'tract_profiles.csv'), index=False)
        return df
