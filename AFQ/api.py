import pandas as pd
import glob
import os.path as op

import numpy as np

import nibabel as nib
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu

import AFQ.data as afd


def do_preprocessing():
    raise NotImplementedError


def make_bundles(bundle_names=["ATR", "CGC", "CST", "FA", "FP", "HCC", "IFO",
                               "ILF", "SLF", "ARC", "UNC"]):
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
                 anat_file="*T1w*"):
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
        self.set_gtab()
        self.set_affine()

    def set_gtab(self):
        self.data_frame['gtab'] = self.data_frame.apply(
            lambda x: dpg.gradient_table(x['bval_file'], x['bvec_file']),
            axis=1)

    def get_gtab(self):
        return self.data_frame['gtab']

    gtab = property(get_gtab, set_gtab)

    def set_affine(self):
        self.data_frame['dwi_affine'] = self.data_frame.apply(
            lambda x: [nib.load(x['dwi_file']).get_affine()], axis=1)

    def get_affine(self):
        return self.data_frame['dwi_affine']

    affine = property(get_affine, set_affine)

    def __getitem__(self, k):
        return self.data_frame.__getitem__(k)

    def set_brain_mask(self, median_radius=4, numpass=4, autocrop=False,
                       vol_idx=None, dilate=None, force_recompute=False):
        self.data_frame['brain_mask_file'] =\
            self.data_frame.apply(_get_fname, suffix='_brain_mask.nii.gz',
                                  axis=1)

        self.data_frame['brain_mask_img'] =\
            self.data_frame.apply(_brain_extract, axis=1,
                                  median_radius=median_radius,
                                  numpass=numpass,
                                  autocrop=autocrop,
                                  vol_idx=vol_idx,
                                  dilate=dilate,
                                  force_recompute=force_recompute)

    def get_brain_mask(self):
        return self.data_frame['brain_mask_img']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def DTI(self, mask=None):
        self.data_frame['dti_params_file'] =\
            self.data_frame.apply(_get_fname,
                                  args=['_dti_params.nii.gz'], axis=1)

        self.data_frame['dti_params_img'] =\
            self.data_fram.apply(_dti)


def _get_fname(row, suffix):
    split_fdwi = op.split(row['dwi_file'])
    fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0] +
                    suffix)
    return fname

def _brain_extract(row, median_radius=4, numpass=4, autocrop=False,
                   vol_idx=None, dilate=None, force_recompute=False):
    if not op.exists(row['brain_mask_file']) or force_recompute:
        img = nib.load(row['dwi_file'])
        data = img.get_data()
        gtab = row['gtab']
        mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
        _, brain_mask = median_otsu(mean_b0, median_radius, numpass,
                                    autocrop, dilate=dilate)
        be_img = nib.Nifti1Image(brain_mask.astype(int),
                                 img.affine)
        nib.save(be_img, row['brain_mask_file'])
        return be_img
    else:
        return nib.load(row['brain_mask_file'])


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
