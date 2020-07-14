import tempfile
import os
import os.path as op
import shutil

import toml

import numpy as np
import numpy.testing as npt

import pandas as pd
from pandas.testing import assert_series_equal

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
import dipy.data as dpd
from dipy.data import fetcher
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.testing.decorators import xvfb_it

from AFQ import api
import AFQ.data as afd
import AFQ.segmentation as seg
import AFQ.utils.streamlines as aus


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def create_dummy_bids_path(n_subjects, n_sessions):
    subjects = ['sub-0%s' % (d + 1) for d in range(n_subjects)]

    # Case where there are individual session folders within each subject's
    # folder:
    if n_sessions > 1:
        sessions = ['ses-0%s' % (d + 1) for d in range(n_sessions)]

        bids_dir = tempfile.mkdtemp()

        afd.to_bids_description(
            bids_dir,
            **{"Name": "Dummy",
               "Subjects": subjects,
               "Sessions": sessions})

        dmriprep_dir = op.join(bids_dir, "derivatives", "dmriprep")
        os.makedirs(dmriprep_dir)
        afd.to_bids_description(
            dmriprep_dir,
            **{"Name": "Dummy",
               "PipelineDescription": {"Name": "synthetic"}})

        for subject in subjects:
            for session in sessions:
                for modality in ['anat', 'dwi']:
                    os.makedirs(
                        op.join(dmriprep_dir, subject, session, modality))
                # Make some dummy data:
                aff = np.eye(4)
                data = np.ones((10, 10, 10, 6))
                bvecs = np.vstack([np.eye(3), np.eye(3)])
                bvecs[0] = 0
                bvals = np.ones(6) * 1000.
                bvals[0] = 0
                np.savetxt(
                    op.join(
                        dmriprep_dir, subject, session, 'dwi', 'dwi.bvals'),
                    bvals)
                np.savetxt(
                    op.join(
                        dmriprep_dir, subject, session, 'dwi', 'dwi.bvecs'),
                    bvecs)
                nib.save(
                    nib.Nifti1Image(data, aff),
                    op.join(
                        dmriprep_dir, subject, session, 'dwi', 'dwi.nii.gz'))
                nib.save(
                    nib.Nifti1Image(data, aff),
                    op.join(
                        dmriprep_dir, subject, session, 'anat', 'T1w.nii.gz'))
                nib.save(
                    nib.Nifti1Image(data, aff),
                    op.join(
                        dmriprep_dir, subject, session, 'anat', 'seg.nii.gz'))
    else:
        # Don't create session folders at all:
        bids_dir = tempfile.mkdtemp()

        afd.to_bids_description(
            bids_dir,
            **{"Name": "Dummy", "Subjects": subjects})

        dmriprep_dir = op.join(bids_dir, "derivatives", "dmriprep")
        os.makedirs(dmriprep_dir)
        afd.to_bids_description(
            dmriprep_dir,
            **{"Name": "Dummy",
               "PipelineDescription": {"Name": "synthetic"}})

        for subject in subjects:
            for modality in ['anat', 'dwi']:
                os.makedirs(op.join(dmriprep_dir, subject, modality))
            # Make some dummy data:
            aff = np.eye(4)
            data = np.ones((10, 10, 10, 6))
            bvecs = np.vstack([np.eye(3), np.eye(3)])
            bvecs[0] = 0
            bvals = np.ones(6) * 1000.
            bvals[0] = 0
            np.savetxt(
                op.join(dmriprep_dir, subject, 'dwi', 'dwi.bvals'),
                bvals)
            np.savetxt(
                op.join(dmriprep_dir, subject, 'dwi', 'dwi.bvecs'),
                bvecs)
            nib.save(
                nib.Nifti1Image(data, aff),
                op.join(dmriprep_dir, subject, 'dwi', 'dwi.nii.gz'))
            nib.save(
                nib.Nifti1Image(data, aff),
                op.join(dmriprep_dir, subject, 'anat', 'T1w.nii.gz'))
            nib.save(
                nib.Nifti1Image(data, aff),
                op.join(dmriprep_dir, subject, 'anat', 'seg.nii.gz'))

    return bids_dir


def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """
    for n_sessions in [1, 2]:
        n_subjects = 3
        bids_path = create_dummy_bids_path(n_subjects, n_sessions)
        my_afq = api.AFQ(bids_path,
                         dmriprep="synthetic",
                         segmentation="synthetic")
        npt.assert_equal(my_afq.data_frame.shape,
                         (n_subjects * n_sessions, 11))


def test_AFQ_data():
    """
    Test if API can run without prealign
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    bids_path = op.join(tmpdir.name, 'stanford_hardi')
    for use_prealign in [True, False]:

        myafq = api.AFQ(
            bids_path=bids_path,
            dmriprep='vistasoft',
            segmentation='freesurfer',
            use_prealign=use_prealign)
        npt.assert_equal(nib.load(myafq.b0[0]).shape,
                         nib.load(myafq['dwi_file'][0]).shape[:3])
        npt.assert_equal(nib.load(myafq.b0[0]).shape,
                         nib.load(myafq.dti[0]).shape[:3])
        myafq.export_rois()


def test_AFQ_slr():
    """
    Test if API can run using slr map
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    bids_path = op.join(tmpdir.name, 'stanford_hardi')
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        segmentation='freesurfer',
        reg_algo='slr',
        moving='subject_sls',
        static='hcp_atlas')
    myafq.export_rois()


# Requires large download
# def test_AFQ_FA():
#     """
#     Test if API can run registeration with FA
#     """
#     tmpdir = nbtmp.InTemporaryDirectory()
#     afd.organize_stanford_data(path=tmpdir.name)
#     myafq = api.AFQ(dmriprep_path=op.join(tmpdir.name, 'stanford_hardi',
#                                           'derivatives', 'dmriprep'),
#                     sub_prefix='sub',
#                     moving='dti_fa_template',
#                     static='dti_fa_subject')
#     myafq.export_rois()


def test_DKI_profile():
    """
    Test using API to profile dki
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_cfin_data(path=tmpdir.name)
    myafq = api.AFQ(bids_path=op.join(tmpdir.name, 'cfin_multib'),
                    dmriprep='dipy')
    myafq.get_dki_fa()
    myafq.get_dki_md()


@xvfb_it
def test_AFQ_data_waypoint():
    """
    Test with some actual data again, this time for track segmentation
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    bids_path = op.join(tmpdir.name, 'stanford_hardi')
    bundle_names = ["SLF", "ARC", "CST", "FP"]
    tracking_params = dict(odf_model="DTI")
    segmentation_params = dict(filter_by_endpoints=False,
                               seg_algo="AFQ",
                               return_idx=True)

    clean_params = dict(return_idx=True)

    myafq = api.AFQ(bids_path=bids_path,
                    dmriprep='vistasoft',
                    segmentation='freesurfer',
                    bundle_names=bundle_names,
                    scalars=["dti_fa", "dti_md"],
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params,
                    clean_params=clean_params,
                    wm_criterion=0.2)

    # Replace the mapping and streamlines with precomputed:
    file_dict = afd.read_stanford_hardi_tractography()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    streamlines = dts.Streamlines(
        dtu.transform_tracking_output(
            [s for s in streamlines if s.shape[0] > 100],
            np.linalg.inv(myafq.dwi_affine[0])))

    sl_file = op.join(
        myafq.data_frame.results_dir[0],
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det_tractography.trk')
    sft = StatefulTractogram(streamlines, myafq.data_frame.dwi_file[0],
                             Space.VOX)
    save_tractogram(sft, sl_file, bbox_valid_check=False)

    mapping_file = op.join(
        myafq.data_frame.results_dir[0],
        'sub-01_ses-01_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz')
    nib.save(mapping, mapping_file)
    reg_prealign_file = op.join(
        myafq.data_frame.results_dir[0],
        'sub-01_ses-01_dwi_prealign_from-DWI_to-MNI_xfm.npy')
    np.save(reg_prealign_file, np.eye(4))

    tgram = load_tractogram(myafq.bundles[0], myafq.dwi_img[0])

    bundles = aus.tgram_to_bundles(tgram, myafq.bundle_dict, myafq.dwi_img[0])
    npt.assert_(len(bundles['CST_R']) > 0)

    # Test ROI exporting:
    myafq.export_rois()
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'ROIs',
        'sub-01_ses-01_dwi_desc-ROI-CST_R-1-include.json'))

    # Test bundles exporting:
    myafq.export_bundles()
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'bundles',
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ-CST_L_tractography.trk'))  # noqa

    # Test creation of file with bundle indices:
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ-clean_tractography_idx.json'))  # noqa

    tract_profiles = pd.read_csv(myafq.tract_profiles[0])
    assert tract_profiles.shape == (400, 5)

    myafq.plot_tract_profiles()
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ_dti_fa_profile_plots.png'))  # noqa

    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ_dti_md_profile_plots.png'))  # noqa

    # Before we run the CLI, we'll remove the bundles and ROI folders, to see
    # that the CLI generates them
    shutil.rmtree(op.join(myafq.data_frame['results_dir'][0],
                          'bundles'))

    shutil.rmtree(op.join(myafq.data_frame['results_dir'][0],
                          'ROIs'))

    # Test the CLI:
    print("Running the CLI:")

    # Bare bones config only points to the files:
    config = dict(files=dict(bids_path=bids_path,
                             dmriprep='vistasoft',
                             segmentation='freesurfer'))

    config_file = op.join(tmpdir.name, "afq_config.toml")
    with open(config_file, 'w') as ff:
        toml.dump(config, ff)

    cmd = "pyAFQ " + config_file
    out = os.system(cmd)
    assert out == 0
    # The combined tract profiles should already exist from the CLI Run:
    from_file = pd.read_csv(op.join(myafq.afq_path, 'tract_profiles.csv'))
    # And should be identical to what we would get by rerunning this:
    combined_profiles = myafq.combine_profiles()
    assert combined_profiles.shape == (400, 7)
    assert_series_equal(combined_profiles['dti_fa'], from_file['dti_fa'])

    # Make sure the CLI did indeed generate these:
    myafq.export_rois()
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'ROIs',
        'sub-01_ses-01_dwi_desc-ROI-CST_R-1-include.json'))

    myafq.export_bundles()
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'bundles',
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ-CST_L_tractography.trk'))  # noqa
