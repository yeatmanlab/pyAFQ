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
        self.set_dwi_affine()

    def set_gtab(self):
        self.data_frame['gtab'] = self.data_frame.apply(
            lambda x: dpg.gradient_table(x['bval_file'], x['bvec_file']),
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

    def _brain_extract(self, row, median_radius=4, numpass=4, autocrop=False,
                       vol_idx=None, dilate=None, force_recompute=False):
        if not op.exists(row['brain_mask_file']) or force_recompute:
            self.set_dwi_data_img()
            img = row['dwi_data_img']
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

    def set_brain_mask(self, median_radius=4, numpass=4, autocrop=False,
                       vol_idx=None, dilate=None, force_recompute=False):
        if 'brain_mask_img' not in self.data_frame.columns or force_recompute:
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(_get_fname, suffix='_brain_mask.nii.gz',
                                      axis=1)

            self.data_frame['brain_mask_img'] =\
                self.data_frame.apply(self._brain_extract,
                                      axis=1,
                                      median_radius=median_radius,
                                      numpass=numpass,
                                      autocrop=autocrop,
                                      vol_idx=vol_idx,
                                      dilate=dilate,
                                      force_recompute=force_recompute)

    def get_brain_mask(self):
        self.set_brain_mask()
        self.data_frame['brain_mask'] =\
            self.data_frame['brain_mask_img'].apply(nib.Nifti1Image.get_data)
        return self.data_frame['brain_mask']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_dwi_data_img(self):
        if 'dwi_data_img' not in self.data_frame.columns:
            self.data_frame['dwi_data_img'] =\
                self.data_frame['dwi_file'].apply(nib.load)

    def get_dwi_data_img(self):
        self.set_dwi_data_img()
        return self.data_frame['dwi_data_img']

    dwi_data_img = property(get_dwi_data_img, set_dwi_data_img)

    def _dti(self, row, mask=None):
        if not op.exists('dti_params_file'):
            img = row['dwi_data_img']
            data = img.get_data()
            gtab = row['gtab']
            if mask is None:
                mask = row['brain_mask_img'].get_data()
            dtf = dti._fit(gtab, data, mask=mask)
            nib.save(nib.Nifti1Image(dtf.model_params, row['dwi_affine']),
                     row['dti_params_file'])
            return dtf
        else:
            model_params = nib.load(row['dti_params_file'])
            gtab = row['gtab']
            tm = TensorModel(gtab)
            return TensorFit(tm, model_params)

    def _dti_params_img(self, row):
        return nib.Nifti1Image(row['dti'].model_params, row['dwi_affine'])

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti']

    def set_dti(self, mask=None):
        self.set_dwi_data_img()
        self.set_brain_mask()
        self.data_frame['dti_params_file'] =\
            self.data_frame.apply(_get_fname, suffix='_dti_params.nii.gz',
                                  axis=1)
        self.data_frame['dti'] = self.data_frame.apply(self._dti,
                                                       axis=1,
                                                       mask=mask)
        self.data_frame['dti_params_img'] =\
            self.data_frame.apply(self._dti_params_img, axis=1)

    dti = property(get_dti, set_dti)

    def get_dti_params(self):
        self.set_dti_params()
        return self.data_frame['dti_params']

    def set_dti_params(self, mask=None):
        self.set_dti()
        self.data_frame['dti_params'] =\
            self.data_frame['dti_params_img'].apply(
                nib.Nifti1Image.get_data)

    dti_params = property(get_dti_params, set_dti_params)


def _get_dti_params(dtf):
    return dti.model_params


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
