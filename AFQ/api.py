# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import pandas as pd
import dask.dataframe as ddf
import glob
import os
import os.path as op
from pathlib import PurePath
import json

import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation


import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu
import dipy.data as dpd
import dipy.tracking.utils as dtu
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.stats.analysis import afq_profile

import AFQ.data as afd
from AFQ.dti import _fit as dti_fit
from AFQ.csd import _fit as csd_fit
import AFQ.tractography as aft
import dipy.reconst.dti as dpy_dti
import AFQ.utils.streamlines as aus
import AFQ.segmentation as seg
import AFQ.registration as reg
import AFQ.utils.volume as auv
from AFQ.utils.bin import get_default_args


__all__ = ["AFQ", "make_bundle_dict"]


def do_preprocessing():
    raise NotImplementedError


BUNDLES = ["ATR", "CGC", "CST", "HCC", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]

DIPY_GH = "https://github.com/nipy/dipy/blob/master/dipy/"


def make_bundle_dict(bundle_names=BUNDLES, seg_algo="afq", resample_to=False):
    """
    Create a bundle dictionary, needed for the segmentation

    Parameters
    ----------
    bundle_names : list, optional
        A list of the bundles to be used in this case. Default: all of them

    resample_to : Nifti1Image, optional
        If set, templates will be resampled to the affine and shape of this
        image.
    """
    if seg_algo == "afq":
        templates = afd.read_templates(resample_to=resample_to)
        callosal_templates = afd.read_callosum_templates(
            resample_to=resample_to)
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

    elif seg_algo == "reco":
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


def _b0(row):
    b0_file = _get_fname(row, '_b0.nii.gz')
    if not op.exists(b0_file):
        img = nib.load(row['dwi_file'])
        data = img.get_fdata()
        gtab = row['gtab']
        mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
        mean_b0_img = nib.Nifti1Image(mean_b0, img.affine)
        nib.save(mean_b0_img, b0_file)
        meta = dict(b0_threshold=gtab.b0_threshold,
                    source=row['dwi_file'])
        meta_fname = _get_fname(row, '_b0.json')
        afd.write_json(meta_fname, meta)
    return b0_file


def _brain_mask(row, median_radius=4, numpass=1, autocrop=False,
                vol_idx=None, dilate=10):
    brain_mask_file = _get_fname(row, '_brain_mask.nii.gz')
    if not op.exists(brain_mask_file):
        b0_file = _b0(row)
        mean_b0_img = nib.load(b0_file)
        mean_b0 = mean_b0_img.get_fdata()
        _, brain_mask = median_otsu(mean_b0, median_radius, numpass,
                                    autocrop, dilate=dilate)
        be_img = nib.Nifti1Image(brain_mask.astype(int),
                                 mean_b0_img.affine)
        nib.save(be_img, brain_mask_file)
        meta = dict(source=b0_file,
                    median_radius=median_radius,
                    numpass=numpass,
                    autocrop=autocrop,
                    vol_idx=vol_idx)
        meta_fname = _get_fname(row, '_brain_mask.json')
        afd.write_json(meta_fname, meta)
    return brain_mask_file


def _dti_fit(row):
    dti_params_file = _dti(row)
    dti_params = nib.load(dti_params_file).get_fdata()
    tm = dpy_dti.TensorModel(row['gtab'])
    tf = dpy_dti.TensorFit(tm, dti_params)
    return tf


def _dti(row):
    dti_params_file = _get_fname(row, '_model-DTI_diffmodel.nii.gz')
    if not op.exists(dti_params_file):
        img = nib.load(row['dwi_file'])
        data = img.get_fdata()
        gtab = row['gtab']
        brain_mask_file = _brain_mask(row)
        mask = nib.load(brain_mask_file).get_fdata()
        dtf = dti_fit(gtab, data, mask=mask)
        nib.save(nib.Nifti1Image(dtf.model_params, row['dwi_affine']),
                 dti_params_file)
        meta_fname = _get_fname(row, '_model-DTI_diffmodel.json')
        meta = dict(
            Parameters=dict(
                FitMethod="WLS"),
            OutlierRejection=False,
            ModelURL=f"{DIPY_GH}reconst/dti.py")
        afd.write_json(meta_fname, meta)
    return dti_params_file


def _csd(row, response=None, sh_order=8, lambda_=1, tau=0.1,):
    csd_params_file = _get_fname(row, '_model-CSD_diffmodel.nii.gz')
    if not op.exists(csd_params_file):
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
        meta_fname = _get_fname(row, '_model-CSD_diffmodel.json')
        meta = dict(SphericalHarmonicDegree=sh_order,
                    ResponseFunctionTensor=response,
                    SphericalHarmonicBasis="DESCOTEAUX",
                    ModelURL=f"{DIPY_GH}reconst/csdeconv.py",
                    lambda_=lambda_,
                    tau=tau)
        afd.write_json(meta_fname, meta)
    return csd_params_file


def _dti_fa(row):
    dti_fa_file = _get_fname(row, '_model-DTI_FA.nii.gz')
    if not op.exists(dti_fa_file):
        tf = _dti_fit(row)
        fa = tf.fa
        nib.save(nib.Nifti1Image(fa, row['dwi_affine']),
                 dti_fa_file)
        meta_fname = _get_fname(row, '_model-DTI_FA.json')
        meta = dict()
        afd.write_json(meta_fname, meta)
    return dti_fa_file


def _dti_cfa(row):
    dti_cfa_file = _get_fname(row, '_model-DTI_desc-DEC_FA.nii.gz')
    if not op.exists(dti_cfa_file):
        tf = _dti_fit(row)
        cfa = tf.color_fa
        nib.save(nib.Nifti1Image(cfa, row['dwi_affine']),
                 dti_cfa_file)
        meta_fname = _get_fname(row, '_model-DTI_desc-DEC_FA.json')
        meta = dict()
        afd.write_json(meta_fname, meta)
    return dti_cfa_file


def _dti_pdd(row):
    dti_pdd_file = _get_fname(row, '_model-DTI_PDD.nii.gz')
    if not op.exists(dti_pdd_file):
        tf = _dti_fit(row)
        pdd = tf.directions.squeeze()
        # Invert the x coordinates:
        pdd[..., 0] = pdd[..., 0] * -1

        nib.save(nib.Nifti1Image(pdd, row['dwi_affine']),
                 dti_pdd_file)
        meta_fname = _get_fname(row, '_model-DTI_PDD.json')
        meta = dict()
        afd.write_json(meta_fname, meta)
    return dti_pdd_file


def _dti_md(row):
    dti_md_file = _get_fname(row, '_model-DTI_MD.nii.gz')
    if not op.exists(dti_md_file):
        tf = _dti_fit(row)
        md = tf.md
        nib.save(nib.Nifti1Image(md, row['dwi_affine']),
                 dti_md_file)
        meta_fname = _get_fname(row, '_model-DTI_MD.json')
        meta = dict()
        afd.write_json(meta_fname, meta)
    return dti_md_file


# Keep track of functions that compute scalars:
_scalar_dict = {"dti_fa": _dti_fa,
                "dti_md": _dti_md}


def _reg_prealign(row):
    prealign_file = _get_fname(row, '_prealign_from-DWI_to-MNI_xfm.npy')
    if not op.exists(prealign_file):
        moving = nib.load(_b0(row))
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
        meta_fname = _get_fname(row, 'prealign_from-DWI_to-MNI_xfm.json')
        meta = dict(type="rigid")
        afd.write_json(meta_fname, meta)
    return prealign_file


def _mapping(row, reg_template):
    mapping_file = _get_fname(row, '_mapping_from-DWI_to_MNI_xfm.nii.gz')
    if not op.exists(mapping_file):
        gtab = row['gtab']
        reg_prealign = np.load(_reg_prealign(row))
        warped_b0, mapping = reg.syn_register_dwi(row['dwi_file'], gtab,
                                                  template=reg_template,
                                                  prealign=reg_prealign)
        mapping.codomain_world2grid = np.linalg.inv(reg_prealign)
        reg.write_mapping(mapping, mapping_file)
        meta_fname = _get_fname(row, '_mapping_reg_prealign.json')
        meta = dict(type="displacementfield")
        afd.write_json(meta_fname, meta)

    return mapping_file


def _wm_mask(row, wm_labels, wm_fa_thresh=0.2):
    wm_mask_file = _get_fname(row, '_wm_mask.nii.gz')
    if not op.exists(wm_mask_file):
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
            meta = dict(source=row['seg_file'],
                        wm_labels=wm_labels)
        else:
            # Otherwise, we'll identify the white matter based on FA:
            fa_fname = _dti_fa(row)
            dti_fa = nib.load(fa_fname).get_fdata()
            wm_mask = dti_fa > wm_fa_thresh
            meta = dict(source=fa_fname,
                        fa_threshold=wm_fa_thresh)

        # Dilate to be sure to reach the gray matter:
        wm_mask = binary_dilation(wm_mask) > 0

        nib.save(nib.Nifti1Image(wm_mask.astype(int), row['dwi_affine']),
                 wm_mask_file)

        meta_fname = _get_fname(row, '_wm_mask.json')
        afd.write_json(meta_fname, meta)

    return wm_mask_file


def _streamlines(row, wm_labels, tracking_params=None):
    """
    wm_labels : list
        The values within the segmentation that are considered white matter. We
        will use this part of the image both to seed tracking (seeding
        throughout), and for stopping.
    """
    if tracking_params is None:
        tracking_params = get_default_args(aft.track)

    odf_model = tracking_params["odf_model"]
    directions = tracking_params["directions"]

    streamlines_file = _get_fname(
        row,
        f'_space-RASMM_model-{odf_model}_desc-{directions}_tractography.trk')

    if not op.exists(streamlines_file):
        if odf_model == "DTI":
            params_file = _dti(row)
        elif odf_model == "CSD":
            params_file = _csd(row)
        wm_mask_fname = _wm_mask(row, wm_labels)
        wm_mask = nib.load(wm_mask_fname).get_fdata().astype(bool)
        tracking_params['seed_mask'] = wm_mask
        tracking_params['stop_mask'] = wm_mask
        sft = aft.track(params_file, **tracking_params)
        sft.to_vox()
        meta_directions = {"det": "deterministic",
                           "prob": "probabilistic"}

        meta = dict(
            TractographyClass="local",
            TractographyMethod=meta_directions[tracking_params["directions"]],
            Count=len(sft.streamlines),
            Seeding=dict(
                ROI=wm_mask_fname,
                n_seeds=tracking_params["n_seeds"],
                random_seeds=tracking_params["random_seeds"]),
            Constraints=dict(AnatomicalImage=wm_mask_fname),
            Parameters=dict(Units="mm",
                            StepSize=tracking_params["step_size"],
                            MinimumLength=tracking_params["min_length"],
                            MaximumLength=tracking_params["max_length"],
                            Unidirectional=False))

        meta_fname = _get_fname(
            row,
            f'_space-RASMM_model-{odf_model}_desc-'
            f'{directions}_tractography.json')
        afd.write_json(meta_fname, meta)
        save_tractogram(sft, streamlines_file, bbox_valid_check=False)

    return streamlines_file


def _segment(row, wm_labels, bundle_dict, reg_template,
             tracking_params, segmentation_params, clean_params):
    # We pass `clean_params` here, but do not use it, so we have the
    # same signature as `_clean_bundles`.
    odf_model = tracking_params["odf_model"]
    directions = tracking_params["directions"]
    seg_algo = segmentation_params["seg_algo"]
    bundles_file = _get_fname(
        row,
        f'_space-RASMM_model-{odf_model}_desc-{directions}-'
        f'{seg_algo}_tractography.trk')

    if not op.exists(bundles_file):
        streamlines_file = _streamlines(
            row,
            wm_labels,
            tracking_params)

        img = nib.load(row['dwi_file'])
        tg = load_tractogram(streamlines_file, img, Space.VOX)
        reg_prealign = np.load(_reg_prealign(row))

        segmentation = seg.Segmentation(**segmentation_params)
        bundles = segmentation.segment(bundle_dict,
                                       tg,
                                       row['dwi_file'],
                                       row['bval_file'],
                                       row['bvec_file'],
                                       reg_template=reg_template,
                                       mapping=_mapping(row, reg_template),
                                       reg_prealign=reg_prealign)

        if segmentation_params['return_idx']:
            idx = {bundle: bundles[bundle]['idx'].tolist()
                   for bundle in bundle_dict}
            afd.write_json(bundles_file.split('.')[0] + '_idx.json',
                           idx)
            bundles = {bundle: bundles[bundle]['sl'] for bundle in bundle_dict}

        tgram = aus.bundles_to_tgram(bundles, bundle_dict, img)
        save_tractogram(tgram, bundles_file)
        meta = dict(source=streamlines_file,
                    Parameters=segmentation_params)
        meta_fname = bundles_file.split('.')[0] + '.json'
        afd.write_json(meta_fname, meta)

    return bundles_file


def _clean_bundles(row, wm_labels, bundle_dict, reg_template, tracking_params,
                   segmentation_params, clean_params):
    odf_model = tracking_params['odf_model']
    directions = tracking_params['directions']
    seg_algo = segmentation_params['seg_algo']
    clean_bundles_file = _get_fname(
        row,
        f'_space-RASMM_model-{odf_model}_desc-{directions}-'
        f'{seg_algo}-clean_tractography.trk')

    if not op.exists(clean_bundles_file):
        bundles_file = _segment(row,
                                wm_labels,
                                bundle_dict,
                                reg_template,
                                tracking_params,
                                segmentation_params,
                                clean_params)

        sft = load_tractogram(bundles_file,
                              row['dwi_img'],
                              Space.VOX)

        tgram = nib.streamlines.Tractogram([], {'bundle': []})
        if clean_params['return_idx']:
            return_idx = {}

        for b in bundle_dict.keys():
            if b != "whole_brain":
                idx = np.where(sft.data_per_streamline['bundle']
                               == bundle_dict[b]['uid'])[0]
                this_tg = StatefulTractogram(
                    sft.streamlines[idx],
                    row['dwi_img'],
                    Space.VOX)
                this_tg = seg.clean_bundle(this_tg, **clean_params)
                if clean_params['return_idx']:
                    this_tg, this_idx = this_tg
                    idx_file = bundles_file.split('.')[0] + '_idx.json'
                    with open(idx_file) as ff:
                        bundle_idx = json.load(ff)[b]
                    return_idx[b] = np.array(bundle_idx)[this_idx].tolist()
                this_tgram = nib.streamlines.Tractogram(
                    this_tg.streamlines,
                    data_per_streamline={
                        'bundle': (len(this_tg)
                                   * [bundle_dict[b]['uid']])},
                        affine_to_rasmm=row['dwi_affine'])
                tgram = aus.add_bundles(tgram, this_tgram)
        save_tractogram(
            StatefulTractogram(tgram.streamlines,
                               sft,
                               Space.VOX,
                               data_per_streamline=tgram.data_per_streamline),
            clean_bundles_file)

        seg_args = get_default_args(seg.clean_bundle)
        for k in seg_args:
            if callable(seg_args[k]):
                seg_args[k] = seg_args[k].__name__

        meta = dict(source=bundles_file,
                    Parameters=seg_args)
        meta_fname = clean_bundles_file.split('.')[0] + '.json'
        afd.write_json(meta_fname, meta)

        if clean_params['return_idx']:
            afd.write_json(clean_bundles_file.split('.')[0] + '_idx.json',
                           return_idx)

    return clean_bundles_file


def _tract_profiles(row, wm_labels, bundle_dict, reg_template,
                    tracking_params, segmentation_params, clean_params,
                    scalars, weighting=None):
    profiles_file = _get_fname(row, '_profiles.csv')
    if not op.exists(profiles_file):
        bundles_file = _clean_bundles(row,
                                      wm_labels,
                                      bundle_dict,
                                      reg_template,
                                      tracking_params,
                                      segmentation_params,
                                      clean_params)
        keys = []
        vals = []
        for k in bundle_dict.keys():
            if k != "whole_brain":
                keys.append(bundle_dict[k]['uid'])
                vals.append(k)
        reverse_dict = dict(zip(keys, vals))

        bundle_names = []
        profiles = []
        node_numbers = []
        scalar_names = []

        trk = nib.streamlines.load(bundles_file)
        for scalar in scalars:
            scalar_file = _scalar_dict[scalar](row)
            scalar_data = nib.load(scalar_file).get_fdata()
            for b in np.unique(trk.tractogram.data_per_streamline['bundle']):
                idx = np.where(
                    trk.tractogram.data_per_streamline['bundle'] == b)[0]
                this_sl = trk.streamlines[idx]
                bundle_name = reverse_dict[b]
                this_profile = afq_profile(
                    scalar_data,
                    this_sl,
                    row["dwi_affine"])
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
        meta = dict(source=bundles_file,
                    parameters=get_default_args(afq_profile))
        meta_fname = profiles_file.split('.')[0] + '.json'
        afd.write_json(meta_fname, meta)

    return profiles_file


def _template_xform(row, reg_template):
    template_xform_file = _get_fname(row, "_template_xform.nii.gz")
    if not op.exists(template_xform_file):
        reg_prealign = np.load(_reg_prealign(row))
        mapping = reg.read_mapping(_mapping(row,
                                            reg_template),
                                   row['dwi_file'],
                                   reg_template,
                                   prealign=np.linalg.inv(reg_prealign))

        template_xform = mapping.transform_inverse(reg_template.get_fdata())
        nib.save(nib.Nifti1Image(template_xform,
                                 row['dwi_affine']),
                 template_xform_file)

    return template_xform_file


def _export_rois(row, bundle_dict, reg_template):
    reg_prealign = np.load(_reg_prealign(row))

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

            warped_roi = auv.patch_up_roi(
                (mapping.transform_inverse(
                    roi.get_fdata(),
                    interpolation='linear')) > 0).astype(int)

            fname = op.split(
                _get_fname(
                    row,
                    f'_desc-ROI-{bundle}-{ii + 1}-{inclusion}.nii.gz'))

            fname = op.join(fname[0], rois_dir, fname[1])

            # Cast to float32, so that it can be read in by MI-Brain:
            nib.save(nib.Nifti1Image(warped_roi.astype(np.float32),
                                     row['dwi_affine']),
                     fname)
            meta = dict()
            meta_fname = fname.split('.')[0] + '.json'
            afd.write_json(meta_fname, meta)


def _export_bundles(row, wm_labels, bundle_dict, reg_template,
                    tracking_params, segmentation_params, clean_params):

    odf_model = tracking_params['odf_model']
    directions = tracking_params['directions']
    seg_algo = segmentation_params['seg_algo']

    for func, folder in zip([_clean_bundles, _segment],
                            ['clean_bundles', 'bundles']):
        bundles_file = func(row,
                            wm_labels,
                            bundle_dict,
                            reg_template,
                            tracking_params,
                            segmentation_params,
                            clean_params)

        bundles_dir = op.join(row['results_dir'], folder)
        os.makedirs(bundles_dir, exist_ok=True)
        trk = nib.streamlines.load(bundles_file)
        tg = trk.tractogram
        streamlines = tg.streamlines
        for bundle in bundle_dict:
            if bundle != "whole_brain":
                uid = bundle_dict[bundle]['uid']
                idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
                this_sl = dtu.transform_tracking_output(
                    streamlines[idx],
                    np.linalg.inv(row['dwi_affine']))

                this_tgm = StatefulTractogram(this_sl, row['dwi_img'],
                                              Space.VOX)

                fname = op.split(
                    _get_fname(
                        row,
                        f'_space-RASMM_model-{odf_model}_desc-{directions}-'
                        f'{seg_algo}-{bundle}_tractography.trk'))
                fname = op.join(fname[0], bundles_dir, fname[1])
                save_tractogram(this_tgm, fname, bbox_valid_check=False)
                meta = dict(source=bundles_file)
                meta_fname = fname.split('.')[0] + '.json'
                afd.write_json(meta_fname, meta)


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
                 sub_prefix="sub",
                 dwi_folder="dwi",
                 dwi_file="*dwi",
                 anat_folder="anat",
                 anat_file="*T1w*",
                 seg_file='*aparc+aseg*',
                 b0_threshold=0,
                 bundle_names=BUNDLES,
                 dask_it=False,
                 reg_template=None,
                 scalars=["dti_fa", "dti_md"],
                 wm_labels=[250, 251, 252, 253, 254, 255, 41, 2, 16, 77],
                 tracking_params=None,
                 segmentation_params=None,
                 clean_params=None):
        """

        dmriprep_path: str
            The path to the preprocessed diffusion data.

        seg_algo : str
            Which algorithm to use for segmentation.
            Can be one of: {"afq", "reco"}

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

        wm_labels : list, optional
            A list of the labels of the white matter in the segmentation file
            used. Default: the white matter values for the segmentation
            provided with the HCP data, including labels for midbrain:
            [250, 251, 252, 253, 254, 255, 41, 2, 16, 77].

        segmentation_params : dict, optional
            The parameters for segmentation. Default: use the default behavior
            of the seg.Segmentation object

        tracking_params: dict, optional
            The parameters for tracking. Default: use the default behavior of
            the aft.track function.
        """

        self.wm_labels = wm_labels

        self.scalars = scalars

        default_tracking_params = get_default_args(aft.track)
        # Replace the defaults only for kwargs for which a non-default value was
        # given:
        if tracking_params is not None:
            for k in tracking_params:
                default_tracking_params[k] = tracking_params[k]

        self.tracking_params = default_tracking_params

        default_seg_params = get_default_args(seg.Segmentation.__init__)
        if segmentation_params is not None:
            for k in segmentation_params:
                default_seg_params[k] = segmentation_params[k]

        self.segmentation_params = default_seg_params
        self.seg_algo = self.segmentation_params["seg_algo"].lower()
        self.bundle_dict = make_bundle_dict(bundle_names=bundle_names,
                                            seg_algo=self.seg_algo,
                                            resample_to=reg_template)

        default_clean_params = get_default_args(seg.clean_bundle)
        if clean_params is not None:
            for k in clean_params:
                default_clean_params[k] = clean_params[k]

        self.clean_params = default_clean_params

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
        if 'b0_file' not in self.data_frame.columns:
            self.data_frame['b0_file'] =\
                self.data_frame.apply(_b0,
                                      axis=1)

    def get_b0(self):
        self.set_b0()
        return self.data_frame['b0_file']

    b0 = property(get_b0, set_b0)

    def set_brain_mask(self):
        if 'brain_mask_file' not in self.data_frame.columns:
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(_brain_mask,
                                      axis=1)

    def get_brain_mask(self):
        self.set_brain_mask()
        return self.data_frame['brain_mask_file']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_wm_mask(self):
        if 'wm_mask_file' not in self.data_frame.columns:
            self.data_frame['wm_mask_file'] =\
                self.data_frame.apply(_wm_mask,
                                      args=[self.wm_labels],
                                      axis=1)

    def get_wm_mask(self):
        self.set_wm_mask()
        return self.data_frame['wm_mask_file']

    wm_mask = property(get_wm_mask, set_wm_mask)

    def set_dti(self):
        if 'dti_params_file' not in self.data_frame.columns:
            self.data_frame['dti_params_file'] =\
                self.data_frame.apply(_dti,
                                      axis=1)

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti_params_file']

    dti = property(get_dti, set_dti)

    def set_dti_fa(self):
        if 'dti_fa_file' not in self.data_frame.columns:
            self.data_frame['dti_fa_file'] =\
                self.data_frame.apply(_dti_fa,
                                      axis=1)

    def get_dti_fa(self):
        self.set_dti_fa()
        return self.data_frame['dti_fa_file']

    dti_fa = property(get_dti_fa, set_dti_fa)

    def set_dti_cfa(self):
        if 'dti_cfa_file' not in self.data_frame.columns:
            self.data_frame['dti_cfa_file'] =\
                self.data_frame.apply(_dti_cfa,
                                      axis=1)

    def get_dti_cfa(self):
        self.set_dti_cfa()
        return self.data_frame['dti_cfa_file']

    dti_cfa = property(get_dti_cfa, set_dti_cfa)

    def set_dti_pdd(self):
        if 'dti_pdd_file' not in self.data_frame.columns:
            self.data_frame['dti_pdd_file'] =\
                self.data_frame.apply(_dti_pdd,
                                      axis=1)

    def get_dti_pdd(self):
        self.set_dti_pdd()
        return self.data_frame['dti_pdd_file']

    dti_pdd = property(get_dti_pdd, set_dti_pdd)

    def set_dti_md(self):
        if 'dti_md_file' not in self.data_frame.columns:
            self.data_frame['dti_md_file'] =\
                self.data_frame.apply(_dti_md,
                                      axis=1)

    def get_dti_md(self):
        self.set_dti_md()
        return self.data_frame['dti_md_file']

    dti_md = property(get_dti_md, set_dti_md)

    def set_mapping(self):
        if 'mapping' not in self.data_frame.columns:
            self.data_frame['mapping'] =\
                self.data_frame.apply(_mapping,
                                      args=[self.reg_template],
                                      axis=1)

    def get_mapping(self):
        self.set_mapping()
        return self.data_frame['mapping']

    mapping = property(get_mapping, set_mapping)

    def set_streamlines(self):
        if 'streamlines_file' not in self.data_frame.columns:
            self.data_frame['streamlines_file'] =\
                self.data_frame.apply(_streamlines, axis=1,
                                      args=[self.wm_labels,
                                            self.tracking_params])

    def get_streamlines(self):
        self.set_streamlines()
        return self.data_frame['streamlines_file']

    streamlines = property(get_streamlines, set_streamlines)

    def set_bundles(self):
        if 'bundles_file' not in self.data_frame.columns:
            self.data_frame['bundles_file'] =\
                self.data_frame.apply(
                    _segment,
                    axis=1,
                    args=[self.wm_labels,
                          self.bundle_dict,
                          self.reg_template,
                          self.tracking_params,
                          self.segmentation_params,
                          self.clean_params])

    def get_bundles(self):
        self.set_bundles()
        return self.data_frame['bundles_file']

    bundles = property(get_bundles, set_bundles)

    def set_clean_bundles(self):
        if 'clean_bundles_file' not in self.data_frame.columns:
            if self.seg_algo == "reco":
                self.data_frame['clean_bundles_file'] =\
                    self.data_frame['bundles_file']
            else:
                self.data_frame['clean_bundles_file'] =\
                    self.data_frame.apply(_clean_bundles, axis=1,
                                          args=[self.wm_labels,
                                                self.bundle_dict,
                                                self.reg_template,
                                                self.tracking_params,
                                                self.segmentation_params,
                                                self.clean_params])

    def get_clean_bundles(self):
        self.set_clean_bundles()
        return self.data_frame['clean_bundles_file']

    clean_bundles = property(get_clean_bundles, set_clean_bundles)

    def set_tract_profiles(self):
        if 'tract_profiles_file' not in self.data_frame.columns:
            self.data_frame['tract_profiles_file'] =\
                self.data_frame.apply(_tract_profiles,
                                      args=[self.wm_labels,
                                            self.bundle_dict,
                                            self.reg_template,
                                            self.tracking_params,
                                            self.segmentation_params,
                                            self.clean_params,
                                            self.scalars],
                                      axis=1)

    def get_tract_profiles(self):
        self.set_tract_profiles()
        return self.data_frame['tract_profiles_file']

    tract_profiles = property(get_tract_profiles, set_tract_profiles)

    def set_template_xform(self):
        if 'template_xform_file' not in self.data_frame.columns:
            self.data_frame['template_xform_file'] = \
                self.data_frame.apply(_template_xform,
                                      args=[self.reg_template],
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
                                    self.tracking_params,
                                    self.segmentation_params,
                                    self.clean_params],
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
