import tempfile
import os
import os.path as op
from AFQ import api

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def create_dummy_preproc_path():
    preproc_dir = tempfile.mkdtemp()
    for subject in ['sub-01', 'sub-02']:
        for session in ['sess01', 'sess02']:
            for modality in ['anat', 'dwi']:
                os.makedirs(op.join(preproc_dir, subject, session, modality))

    for subject in ['sub-01', 'sub-02']:
        for session in ['sess01', 'sess02']:
            touch(op.join(preproc_dir, subject, session,'anat', 'T1w.nii.gz'))
            touch(op.join(preproc_dir, subject, session,'dwi', 'dwi.nii.gz'))
            touch(op.join(preproc_dir, subject, session,'dwi', 'dwi.bvals'))
            touch(op.join(preproc_dir, subject, session,'dwi', 'dwi.bvecs'))

    return preproc_dir

def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """

    preproc_path = create_dummy_preproc_path()
    my_afq = api.AFQ(preproc_path=preproc_path)
