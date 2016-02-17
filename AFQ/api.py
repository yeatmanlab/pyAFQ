import pandas as pd
import glob
import os.path as op
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu


def do_preprocessing():
    raise NotImplementedError


class AFQ(object):
    """
    Requires the following file structure:

    study_folder/
                sub-01
                sub-02/sess-test
                sub-02/sess-retest/anat/T1w.nii.gz
                sub-02/sess-retest/dwi/dwi.nii.gz
                sub-02/sess-retest/dwi/dwi.bvals
                sub-02/sess-retest/dwi/dwi.bvecs


    All subjects'/sessions' dwi_prefix needs to be the same!

    That is, you *can't* have:

        sub-01/sess-test/dwi/dwi.nii.gz
        sub-02/sess-retest/dwi/dwi_foo.nii.gz

    Even though these are different subjects/sessions/etc.

    Instead you can (and must!) have:

        sub-01/sess-test/dwi/mydata_foo.nii.gz
        sub-02/sess-retest/dwi/mydata_foo.nii.gz

    Where any part of this string (e.g. "foo") can report on things like the
    number of directions, the preprocessing that happened, etc.

    """
    def __init__(self, raw_path=None, preproc_path=None,
                 sub_prefix="sub", dwi_folder="dwi",
                 dwi_file="*dwi*", anat_folder="anat",
                 anat_file="*T1w"):
        self.raw_path = raw_path
        self.preproc_path = preproc_path
        if self.preproc_path is None:
            if self.raw_path is None:
                e_s = "must provide either preproc_path or raw_path (or both)"
                raise ValueError(e_s)
            # This creates the preproc_path such that:
            self.preproc_path = do_preprocessing(self.raw_path)
        # This is the place in which each subject's full data lives
        self.subject_dirs = glob.glob(preproc_path + '%s*'%sub_prefix)
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
                dwi_file_list.append(glob.glob(op.join(sub_dir, '%s/%s/%s.nii.gz'%(sess, dwi_folder, dwi_file)))[0])
                bvec_file_list.append(glob.glob(op.join(sub_dir, '%s/%s/%s.bvec*'%(sess, dwi_folder, dwi_file)))[0])
                bval_file_list.append(glob.glob(op.join(sub_dir, '%s/%s/%s.bval*'%(sess, dwi_folder, dwi_file)))[0])
                anat_file_list.append(glob.glob(op.join(sub_dir, '%s/%s/%s.nii.gz'%(sess, anat_folder, anat_file)))[0])
                sub_list.append(subject)
                sess_list.append(sess)

        self.data_frame = pd.DataFrame(dict(subjects=sub_list,
                                            dwi_file=dwi_file_list,
                                            bvec_file=bvec_file_list,
                                            bval_file=bval_file_list,
                                            anat_file=anat_file_list,
                                            sess=sess_list))
        self.init_gtab()
        self.init_affine()

    #def compute_profiles(self, force_recompute=False)

    def init_affine(self):
        affine_list = []
        for fdwi in self.data_frame['dwi_file']:
            affine_list.append(nib.load().get_affine())
        self.data_frame['dwi_affine'] = affine_list

    def init_gtab(self):
        gtab_list = []
        for fbval, fbvec in zip(self.data_frame['bval_file'],
                                self.data_frame['bvec_file']):
            gtab_list.append(dpg.gradient_table(fbval, fbvec))
        self.data_frame['gtab'] = gtab_list


    def brain_extraction(self, median_radius=4, numpass=4, autocrop=False,
                         vol_idx=None, dilate=None, force_recompute=False):

        if 'brain_mask' in self.data_frame.columns:
            if not force_recompute:
                return

        self.data_frame['brain_mask'] = np.nan
        if 'brain_mask_file' in self.data_frame.columns:
            if not force_recompute:
                for be_fname in self.data_frame['brain_mask_file']:
                    self.data_frame['brain_mask'] =\
                                nib.load(be_fname).get_data()
                    return
        self.data_frame['brain_mask_file'] = np.nan

        for gtab, fdwi in zip(self.data_frame['gtab'],
                              self.data_frame['dwi_file']):
            split_fdwi = op.split(fdwi)
            be_fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0],
                               '_brain_mask.nii.gz')
            self.data_frame['brain_mask_file'][
                    self.data_frame['dwi_file'] == fdwi] = be_fname

            if not op.exists(be_fname):
                _, brain_mask = median_otsu(mean_b0, median_radius, numpass,
                                            autocrop, dilate=dilate)
                nib.save(nib.Nifti1Image(brain_mask.astype(int), img.affine),
                         be_fname)

            else:
                self.data_frame['brain_mask'] =\
                 nib.load(be_fname).get_data()
