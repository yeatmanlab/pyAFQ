import pandas as pd
import glob
import os.path as op

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
    def fit_tensors():
        raise NotImplementedError
