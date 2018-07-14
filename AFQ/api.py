# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import pandas as pd
import dask.dataframe as ddf
import glob
import os.path as op

import numpy as np

import nibabel as nib

import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu
import dipy.data as dpd

import AFQ.data as afd
from AFQ.dti import _fit as dti_fit
import AFQ.tractography as aft
import dipy.reconst.dti as dpy_dti
import AFQ.utils.streamlines as aus
import AFQ.segmentation as seg
import AFQ.registration as reg


def do_preprocessing():
    raise NotImplementedError


BUNDLES = ["ATR", "CGC", "CST", "HCC", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]


def make_bundle_dict(bundle_names=BUNDLES):
    """
    Create a bundle dictionary, needed for the segmentation

    Parameters
    ----------
    bundle_names : list, optional
        A list of the bundles to be used in this case. Default: all of them
    """
    templates = afd.read_templates()
    # For the arcuate, we need to rename a few of these and duplicate the SLF
    # ROI:
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
                         templates[name + "_R"]],
                'rules': [True, True],
                'prob_map': templates[name + "_prob_map"],
                'cross_midline': True,
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

    return afq_bundles


def _brain_mask(row, median_radius=4, numpass=4, autocrop=False,
                vol_idx=None, dilate=None, force_recompute=False):
    brain_mask_file = _get_fname(row, '_brain_mask.nii.gz')
    if not op.exists(brain_mask_file) or force_recompute:
        img = nib.load(row['dwi_file'])
        data = img.get_data()
        gtab = row['gtab']
        mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
        _, brain_mask = median_otsu(mean_b0, median_radius, numpass,
                                    autocrop, dilate=dilate)
        be_img = nib.Nifti1Image(brain_mask.astype(int),
                                 img.affine)
        nib.save(be_img, brain_mask_file)
    return brain_mask_file


def _dti(row, force_recompute=False):
    dti_params_file = _get_fname(row, '_dti_params.nii.gz')
    if not op.exists(dti_params_file) or force_recompute:
        img = nib.load(row['dwi_file'])
        data = img.get_data()
        gtab = row['gtab']
        brain_mask_file = _brain_mask(row)
        mask = nib.load(brain_mask_file).get_data()
        dtf = dti_fit(gtab, data, mask=mask)
        nib.save(nib.Nifti1Image(dtf.model_params, row['dwi_affine']),
                 dti_params_file)
    return dti_params_file


def _dti_fit(row):
    dti_params_file = _dti(row)
    dti_params = nib.load(dti_params_file).get_data()
    tm = dpy_dti.TensorModel(row['gtab'])
    tf = dpy_dti.TensorFit(tm, dti_params)
    return tf


def _dti_fa(row, force_recompute=False):
    dti_fa_file = _get_fname(row, '_dti_fa.nii.gz')
    if not op.exists(dti_fa_file) or force_recompute:
        tf = _dti_fit(row)
        fa = tf.fa
        nib.save(nib.Nifti1Image(fa, row['dwi_affine']),
                 dti_fa_file)
    return dti_fa_file


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


def _mapping(row, force_recompute=False):
    mapping_file = _get_fname(row, '_mapping.nii.gz')
    if not op.exists(mapping_file) or force_recompute:
        gtab = row['gtab']
        reg_template = dpd.read_mni_template()
        mapping = reg.syn_register_dwi(row['dwi_file'], gtab,
                                       template=reg_template)

        reg.write_mapping(mapping, mapping_file)
    return mapping_file


def _streamlines(row, wm_labels, odf_model="DTI", directions="det", seeds=2,
                 force_recompute=False):
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
        else:
            raise(NotImplementedError)

        seg_img = nib.load(row['seg_file'])
        dwi_img = nib.load(row['dwi_file'])
        seg_data_orig = seg_img.get_data()

        # For different sets of labels, extract all the voxels that have any
        # of these values:
        wm_mask = np.sum(np.concatenate([(seg_data_orig == l)[..., None]
                                         for l in wm_labels], -1), -1)

        dwi_data = dwi_img.get_data()
        resamp_wm = np.round(reg.resample(wm_mask, dwi_data[..., 0],
                             seg_img.affine,
                             dwi_img.affine)).astype(int)

        streamlines = aft.track(params_file,
                                directions='det',
                                seeds=seeds,
                                seed_mask=resamp_wm,
                                stop_mask=resamp_wm)

        aus.write_trk(streamlines_file, streamlines,
                      affine=row['dwi_affine'])

    return streamlines_file


def _bundles(row, wm_labels, odf_model="DTI", directions="det", seeds=2,
             force_recompute=False):
    bundles_file = _get_fname(row,
                              '%s_%s_bundles.trk' % (odf_model,
                                                     directions))
    if not op.exists(bundles_file) or force_recompute:
        streamlines_file = _streamlines(row, wm_labels,
                                        odf_model=odf_model,
                                        directions=directions,
                                        seeds=seeds,
                                        force_recompute=force_recompute)
        tg = nib.streamlines.load(streamlines_file).tractogram
        sl = tg.apply_affine(np.linalg.inv(row['dwi_affine'])).streamlines
        bundle_dict = make_bundle_dict()
        reg_template = dpd.read_mni_template()
        mapping = reg.read_mapping(_mapping(row), row['dwi_file'],
                                   reg_template)
        bundles = seg.segment(row['dwi_file'],
                              row['bval_file'],
                              row['bvec_file'],
                              sl,
                              bundle_dict,
                              reg_template=reg_template,
                              mapping=mapping)
        tgram = aus.bundles_to_tgram(bundles, bundle_dict, row['dwi_affine'])
        nib.streamlines.save(tgram, bundles_file)
    return bundles_file


def _clean_bundles(row, wm_labels, odf_model="DTI", directions="det", seeds=2,
             force_recompute=False):
    clean_bundles_file = _get_fname(row,
                                    '%s_%s_clean_bundles.trk' % (odf_model,
                                                                 directions))
    if not op.exists(clean_bundles_file) or force_recompute:
        bundles_file = _bundles(row,
                                wm_labels,
                                odf_model="DTI",
                                directions="det",
                                seeds=seeds,
                                force_recompute=False)
        tg = nib.streamlines.load(bundles_file).tractogram
        sl = tg.apply_affine(np.linalg.inv(row['dwi_affine'])).streamlines
        bundle_dict = make_bundle_dict()
        tgram = nib.streamlines.Tractogram([], {'bundle': []})
        for b in bundle_dict.keys():
            idx = np.where(tg.data_per_streamline['bundle'] ==
                           bundle_dict[b]['uid'])[0]
            this_sl = sl[idx]
            this_sl = seg.clean_fiber_group(this_sl)
            this_tgram = nib.streamlines.Tractogram(
                this_sl,
                data_per_streamline={
                    'bundle': (len(this_sl) *
                               [bundle_dict[b]['uid']])},
                    affine_to_rasmm=row['dwi_affine'])
            tgram = aus.add_bundles(tgram, this_tgram)
        nib.streamlines.save(tgram, clean_bundles_file)

    return bundles_file



def _tract_profiles(row, wm_labels, odf_model="DTI", directions="det",
                    scalars=["dti_fa", "dti_md"], weighting=None,
                    force_recompute=False):
    profiles_file = _get_fname(row, '_profiles.csv')
    if not op.exists(profiles_file) or force_recompute:
        bundles_file = _bundles(row,
                                wm_labels,
                                odf_model=odf_model,
                                directions=directions,
                                force_recompute=force_recompute)
        bundle_dict = make_bundle_dict()
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
            scalar_data = nib.load(scalar_file).get_data()
            for b in np.unique(trk.tractogram.data_per_streamline['bundle']):
                idx = np.where(
                    trk.tractogram.data_per_streamline['bundle'] == b)[0]
                this_sl = list(trk.streamlines[idx])
                bundle_name = reverse_dict[b]
                this_profile = seg.calculate_tract_profile(
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


class AFQ(object):
    """
    This is file folder structure that AFQ requires in your study folder::

        ├── sub01
        │   ├── sess01
        │   │   ├── anat
        │   │   │   ├── sub-01_sess-01_aparc+aseg.nii.gz
        │   │   │   └── sub-01_sess-01_T1.nii.gz
        │   │   └── dwi
        │   │       ├── sub-01_sess-01_dwi.bvals
        │   │       ├── sub-01_sess-01_dwi.bvecs
        │   │       └── sub-01_sess-01_dwi.nii.gz
        │   └── sess02
        │       ├── anat
        │       │   ├── sub-01_sess-02_aparc+aseg.nii.gz
        │       │   └── sub-01_sess-02_T1w.nii.gz
        │       └── dwi
        │           ├── sub-01_sess-02_dwi.bvals
        │           ├── sub-01_sess-02_dwi.bvecs
        │           └── sub-01_sess-02_dwi.nii.gz
        └── sub02
            ├── sess01
            │   ├── anat
            │       ├── sub-02_sess-01_aparc+aseg.nii.gz
            │   │   └── sub-02_sess-01_T1w.nii.gz
            │   └── dwi
            │       ├── sub-02_sess-01_dwi.bvals
            │       ├── sub-02_sess-01_dwi.bvecs
            │       └── sub-02_sess-01_dwi.nii.gz
            └── sess02
                ├── anat
                │   ├── sub-02_sess-02_aparc+aseg.nii.gz
                │   └── sub-02_sess-02_T1w.nii.gz
                └── dwi
                    ├── sub-02_sess-02_dwi.bvals
                    ├── sub-02_sess-02_dwi.bvecs
                    └── sub-02_sess-02_dwi.nii.gz

    For now, it is up to users to arrange this file folder structure in their
    data, with preprocessed data, but in the future, this structure will be
    automatically generated from BIDS-compliant preprocessed data [1]_.

    Notes
    -----
    The structure of the file-system required here resembles that specified
    by BIDS [1]_. In the future, this will be organized according to the
    BIDS derivatives specification, as we require preprocessed, rather than
    raw data.

    .. [1] Gorgolewski et al. (2016). The brain imaging data structure,
           a format for organizing and describing outputs of neuroimaging
           experiments. Scientific Data, 3::160044. DOI: 10.1038/sdata.2016.44.

    """
    def __init__(self, raw_path=None, preproc_path=None,
                 sub_prefix="sub", dwi_folder="dwi",
                 dwi_file="*dwi", anat_folder="anat",
                 anat_file="*T1w*", seg_file='*aparc+aseg*',
                 b0_threshold=0, odf_model="DTI", directions="det", seeds=2,
                 bundle_list=BUNDLES, dask_it=False,
                 force_recompute=False,
                 wm_labels=[251, 252, 253, 254, 255, 41, 2]):
        """

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
            provided with the HCP data: [251, 252, 253, 254, 255, 41, 2].
        """
        self.directions = directions
        self.odf_model = odf_model
        self.raw_path = raw_path
        self.bundle_list = bundle_list
        self.force_recompute = force_recompute
        self.wm_labels = wm_labels
        self.seeds = seeds

        self.preproc_path = preproc_path
        if self.preproc_path is None:
            if self.raw_path is None:
                e_s = "must provide either preproc_path or raw_path (or both)"
                raise ValueError(e_s)
            # This creates the preproc_path such that everything else works:
            self.preproc_path = do_preprocessing(self.raw_path)
        # This is the place in which each subject's full data lives
        self.subject_dirs = glob.glob(op.join(preproc_path,
                                              '%s*' % sub_prefix))
        self.subjects = [op.split(p)[-1] for p in self.subject_dirs]
        sub_list = []
        sess_list = []
        dwi_file_list = []
        bvec_file_list = []
        bval_file_list = []
        anat_file_list = []
        seg_file_list = []
        for subject, sub_dir in zip(self.subjects, self.subject_dirs):
            sessions = glob.glob(op.join(sub_dir, '*'))
            for sess in sessions:
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

                anat_file_list.append(glob.glob(op.join(sub_dir,
                                                        ('%s/%s/%s.nii.gz' %
                                                         (sess,
                                                          anat_folder,
                                                          anat_file))))[0])

                seg_file_list.append(glob.glob(op.join(sub_dir,
                                                       ('%s/%s/%s.nii.gz' %
                                                        (sess,
                                                         anat_folder,
                                                         seg_file))))[0])

                sub_list.append(subject)
                sess_list.append(sess)

        self.data_frame = pd.DataFrame(dict(subject=sub_list,
                                            dwi_file=dwi_file_list,
                                            bvec_file=bvec_file_list,
                                            bval_file=bval_file_list,
                                            anat_file=anat_file_list,
                                            seg_file=seg_file_list,
                                            sess=sess_list))
        if dask_it:
            self.data_frame = ddf.from_pandas(self.data_frame,
                                              npartitions=len(sub_list))
        self.set_gtab(b0_threshold)
        self.set_dwi_affine()

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

    def __getitem__(self, k):
        return self.data_frame.__getitem__(k)

    def set_brain_mask(self, median_radius=4, numpass=4, autocrop=False,
                       vol_idx=None, dilate=None):
        if ('brain_mask_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(_brain_mask,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_brain_mask(self):
        self.set_brain_mask()
        return self.data_frame['brain_mask_file']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_dti(self):
        if ('dti_params_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['dti_params_file'] =\
                self.data_frame.apply(_dti,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti_params_file']

    dti = property(get_dti, set_dti)

    def set_dti_fa(self):
        if ('dti_fa_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['dti_fa_file'] =\
                self.data_frame.apply(_dti_fa,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_fa(self):
        self.set_dti_fa()
        return self.data_frame['dti_fa_file']

    dti_fa = property(get_dti_fa, set_dti_fa)

    def set_dti_md(self):
        if ('dti_md_file' not in self.data_frame.columns or
                self.force_recompute):
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
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_mapping(self):
        self.set_mapping()
        return self.data_frame['mapping']

    mapping = property(get_mapping, set_mapping)

    def set_streamlines(self):
        if ('streamlines_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['streamlines_file'] =\
                self.data_frame.apply(_streamlines, axis=1,
                                      args=[self.wm_labels],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      seeds=self.seeds,
                                      force_recompute=self.force_recompute)

    def get_streamlines(self):
        self.set_streamlines()
        return self.data_frame['streamlines_file']

    streamlines = property(get_streamlines, set_streamlines)

    def set_bundles(self):
        if ('bundles_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['bundles_file'] =\
                self.data_frame.apply(_bundles, axis=1,
                                      args=[self.wm_labels],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      seeds=self.seeds,
                                      force_recompute=self.force_recompute)

    def get_bundles(self):
        self.set_bundles()
        return self.data_frame['bundles_file']

    bundles = property(get_bundles, set_bundles)

    def set_clean_bundles(self):
        if ('clean_bundles_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['clean_bundles_file'] =\
                self.data_frame.apply(_clean_bundles, axis=1,
                                      args=[self.wm_labels],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      seeds=self.seeds,
                                      force_recompute=self.force_recompute)

    def get_clean_bundles(self):
        self.set_clean_bundles()
        return self.data_frame['clean_bundles_file']

    clean_bundles = property(get_clean_bundles, set_clean_bundles)

    def set_tract_profiles(self):
        if ('tract_profiles_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['tract_profiles_file'] =\
                self.data_frame.apply(_tract_profiles,
                                      args=[self.wm_labels],
                                      force_recompute=self.force_recompute,
                                      axis=1)

    def get_tract_profiles(self):
        self.set_tract_profiles()
        return self.data_frame['tract_profiles_file']

    tract_profiles = property(get_tract_profiles, set_tract_profiles)


def _get_affine(fname):
    return nib.load(fname).get_affine()


def _get_fname(row, suffix):
    split_fdwi = op.split(row['dwi_file'])
    fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0] +
                    suffix)
    return fname
