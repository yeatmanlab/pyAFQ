import tempfile
import os
import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.tracking.utils as dtu

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



def test_AFQ_data():
    """
    Test with some actual data
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    file_dict = afd.read_stanford_hardi_tractography()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    streamlines = [s for s in streamlines if s.shape[0] > 10]
    myafq = api.AFQ(
        preproc_path=op.join(tmpdir.name, 'stanford_hardi'),
        sub_prefix='sub')

    # For things to go reasonably fast, we'll set a brain mask that
    # includes only a small number of voxels and use that:
    streamlines = list(dtu.move_streamlines(
                            streamlines,
                            np.linalg.inv(myafq.dwi_affine[0])))

    dwi_shape = nib.load(myafq.data_frame.dwi_file[0]).shape
    brain_mask = np.zeros(dwi_shape[:-1])
    for sl in streamlines:
        brain_mask[sl.astype(int)] = 1
    nib.save(nib.Nifti1Image(brain_mask, myafq.dwi_affine[0]),
            op.join(tmpdir.name, 'brain_mask.nii.gz'))

    # Replace the brain-mask before moving on:
    myafq.brain_mask[0] = op.join(tmpdir.name, 'brain_mask.nii.gz')
    1/0.
    myafq.bundles