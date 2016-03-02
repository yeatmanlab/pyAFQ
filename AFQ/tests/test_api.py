import tempfile
import os
import os.path as op

import numpy.testing as npt

from AFQ import api


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def create_dummy_preproc_path(n_subjects, n_sessions):
    preproc_dir = tempfile.mkdtemp()
    subjects = ['sub-%s' % (d + 1) for d in range(n_subjects)]
    sessions = ['sess-%s' % (d + 1) for d in range(n_sessions)]
    for subject in subjects:
        for session in sessions:
            for modality in ['anat', 'dwi']:
                os.makedirs(op.join(preproc_dir, subject, session, modality))
            touch(op.join(preproc_dir, subject, session, 'anat', 'T1w.nii.gz'))
            touch(op.join(preproc_dir, subject, session, 'dwi', 'dwi.nii.gz'))
            touch(op.join(preproc_dir, subject, session, 'dwi', 'dwi.bvals'))
            touch(op.join(preproc_dir, subject, session, 'dwi', 'dwi.bvecs'))

    return preproc_dir


def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """
    n_subjects = 3
    n_sessions = 2
    preproc_path = create_dummy_preproc_path(n_subjects, n_sessions)
    my_afq = api.AFQ(preproc_path=preproc_path)
    npt.assert_equal(my_afq.shape, (n_subjects * n_sessions, 6))
