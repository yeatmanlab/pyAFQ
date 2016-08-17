# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os.path as op

import numpy as np

import nibabel as nib
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu

import AFQ.data as afd
import AFQ.dti as dti
from dipy.reconst.dti import TensorModel, TensorFit


def do_preprocessing():
    raise NotImplementedError


def make_bundle_dict(bundle_names=["ATR", "CGC", "CST", "FA", "FP",
                                   "HCC", "IFO", "ILF", "SLF", "ARC", "UNC"]):
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
    for name in bundle_names:
        for hemi in ['_R', '_L']:
            afq_bundles[name + hemi] = {'ROIs': [templates[name + '_roi1' +
                                                           hemi],
                                                 templates[name + '_roi1' +
                                                           hemi]],
                                        'rules': [True, True]}


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
        dtf = dti._fit(gtab, data, mask=mask)
        nib.save(nib.Nifti1Image(dtf.model_params, row['dwi_affine']),
                 dti_params_file)
    return dti_params_file

def _streamlines(row, force_recompute=False):
    streamlines_file = _get_fname(row, '_streamlines.trk')
    if not op.exists(row['streamlines_file']) or force_recompute:
        dti_params_file = _dti(row)
        streamlines = list(aft.track(dti_params_file))
        aus.write_trk(streamlines_file, streamlines,
                      affine=row['dwi_affine'])
    return streamlines_file


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
    automatically generated from

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
                 dwi_file="*dwi*", anat_folder="anat",
                 anat_file="*T1w*", b0_threshold=0):
        self.raw_path = raw_path
        self.preproc_path = preproc_path
        if self.preproc_path is None:
            if self.raw_path is None:
                e_s = "must provide either preproc_path or raw_path (or both)"
                raise ValueError(e_s)
            # This creates the preproc_path such that:
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
                sub_list.append(subject)
                sess_list.append(sess)

        self.data_frame = pd.DataFrame(dict(subject=sub_list,
                                            dwi_file=dwi_file_list,
                                            bvec_file=bvec_file_list,
                                            bval_file=bval_file_list,
                                            anat_file=anat_file_list,
                                            sess=sess_list))
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
                       vol_idx=None, dilate=None, force_recompute=False):
        if 'brain_mask_file' not in self.data_frame.columns or force_recompute:
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(_brain_mask, axis=1)

    def get_brain_mask(self):
        self.set_brain_mask()
        return self.data_frame['brain_mask_file']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_dti(self, force_recompute=False):
        if 'dti_params_file' not in self.data_frame.columns or force_recompute:
            self.data_frame['dti_params_file'] =\
                self.data_frame.apply(_dti, axis=1)

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti_params_file']

    dti = property(get_dti, set_dti)

    def set_streamlines(self, force_recompute=False):
        if 'streamlines_file' not in self.data_frame.columns or force_recompute:
            self.data_frame['streamlines_file'] =\
                self.data_frame.apply(_streamlines, axis=1)

    def get_streamlines(self):
        self.set_streamlines()
        return self.data_frame['streamlines_file']


def _get_affine(fname):
    return nib.load(fname).get_affine()


def _get_fname(row, suffix):
    split_fdwi = op.split(row['dwi_file'])
    fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0] +
                    suffix)
    return fname

# def _tensor_fa_fname(row):
#     split_fdwi = op.split(row['dwi_file'])
#     be_fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0] +
#                        'dti_FA.nii.gz')
#
#
# def _tensor_fnames(row):
#     split_fdwi = op.split(row['dwi_file'])
#     names = ['FA', 'MD', 'AD', 'RD', 'params']
#
#     for n in names:
#         file_paths[n] = op.join(out_dir, 'dti_%s.nii.gz' % n)
#         be_fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0] +
#                            '_brain_mask.nii.gz')
#
#
# def _fit_tensor(row, mask=None, force_recompute):
#     if not op.exists(row['dti_files']) or force_recompute:
#         out_dir = op.split(row['dwi_file'])[0]
#         dt_files = fit_dti(row['dwi_file'], row['bval_file'],
#                            row['bvec_file'], mask=mask,
#                            out_dir=out_dir, b0_threshold=b0_threshold)
