import tempfile
import os
import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

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
            nib.save(nib.Nifti1Image(data, aff),
                     op.join(preproc_dir, subject, session, 'anat',
                             'aparc+aseg.nii.gz'))

    return preproc_dir


def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """
    n_subjects = 3
    n_sessions = 2
    preproc_path = create_dummy_preproc_path(n_subjects, n_sessions)
    my_afq = api.AFQ(preproc_path=preproc_path)
    npt.assert_equal(my_afq.data_frame.shape, (n_subjects * n_sessions, 9))


def test_AFQ_data():
    """
    Test with some actual data
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    myafq = api.AFQ(preproc_path=op.join(tmpdir.name, 'stanford_hardi'),
                    sub_prefix='sub')
    npt.assert_equal(nib.load(myafq.brain_mask[0]).shape,
                     nib.load(myafq['dwi_file'][0]).shape[:3])
    npt.assert_equal(nib.load(myafq.brain_mask[0]).shape,
                     nib.load(myafq.dti[0]).shape[:3])
