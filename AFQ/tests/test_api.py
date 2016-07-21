import tempfile
import os
import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
from AFQ import api
import AFQ.data as afd


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
            # Make some dummy data:
            aff = np.eye(4)
            data = np.ones((10, 10, 10, 6))
            bvecs = np.vstack([np.eye(3), np.eye(3)])
            bvecs[0] = 0
            bvals = np.ones(6) * 1000.
            bvals[0] = 0
            np.savetxt(op.join(preproc_dir, subject, session, 'dwi',
                               'dwi.bvals'),
                       bvals)
            np.savetxt(op.join(preproc_dir, subject, session, 'dwi',
                               'dwi.bvecs'),
                       bvecs)
            nib.save(nib.Nifti1Image(data, aff),
                     op.join(preproc_dir, subject, session, 'dwi',
                             'dwi.nii.gz'))
            nib.save(nib.Nifti1Image(data, aff),
                     op.join(preproc_dir, subject, session, 'anat',
                             'T1w.nii.gz'))

    return preproc_dir


def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """
    n_subjects = 3
    n_sessions = 2
    preproc_path = create_dummy_preproc_path(n_subjects, n_sessions)
    my_afq = api.AFQ(preproc_path=preproc_path)
    npt.assert_equal(my_afq.data_frame.shape, (n_subjects * n_sessions, 8))


def test_AFQ_data():
    """
    Test with some actual data
    """
    afd.organize_stanford_data()
    myafq = api.AFQ(preproc_path=op.join(afd.afq_home, 'stanford_hardi'),
                    sub_prefix='sub')
    myafq.set_brain_mask()
