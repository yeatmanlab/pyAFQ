import tempfile
import os
import os.path as op
import shutil

import toml

import numpy as np
import numpy.testing as npt
import pytest

import pandas as pd
from pandas.testing import assert_series_equal

from bids.exceptions import BIDSValidationError

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
import dipy.data as dpd
from dipy.data import fetcher, get_fnames
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.testing.decorators import xvfb_it

from AFQ import api
import AFQ.data as afd
import AFQ.segmentation as seg
import AFQ.utils.streamlines as aus
import AFQ.registration as reg
import AFQ.utils.bin as afb
from AFQ.mask import RoiMask, ThresholdedScalarMask, PFTMask, MaskFile


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def get_temp_hardi():
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    bids_path = op.join(tmpdir.name, 'stanford_hardi')

    sub_path = op.join(
        tmpdir.name,
        'stanford_hardi',
        'derivatives',
        'vistasoft',
        'sub-01',
        'ses-01',
        'dwi')

    return tmpdir, bids_path, sub_path


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


@pytest.mark.nightly4
def test_AFQ_custom_tract():
    """
    Test whether AFQ can use tractography from
    custom_tractography_bids_filters
    """
    _, bids_path, sub_path = get_temp_hardi()
    afd.fetch_stanford_hardi_tractography()

    bundle_names = ["SLF", "ARC", "CST", "FP"]

    # move subsampled tractography into bids folder
    os.rename(
        op.join(
            op.expanduser('~'),
            'AFQ_data',
            'stanford_hardi_tractography',
            'tractography_subsampled.trk'),
        op.join(
            sub_path,
            'subsampled_tractography.trk'
            )
    )
    my_afq = api.AFQ(
        bids_path,
        dmriprep='vistasoft',
        bundle_info=bundle_names,
        custom_tractography_bids_filters={
            "suffix": "tractography",
            "scope": "vistasoft"
        })
    my_afq.export_rois()


@pytest.mark.nightly2
def test_AFQ_no_derivs():
    """
    Test the initialization of the AFQ object
    """
    bids_path = create_dummy_bids_path(1, 1)
    os.remove(op.join(
        bids_path, "derivatives", "dmriprep", "dataset_description.json"))
    with pytest.raises(
            ValueError,
            match="`bids_path` contains no subjects in derivatives folders."
                  + " This could be caused by derivatives folders not"
                  + " following the BIDS format."):
        my_afq = api.AFQ(bids_path,
                         dmriprep="synthetic")


@pytest.mark.nightly4
@xvfb_it
def test_AFQ_fury():
    _, bids_path, _ = get_temp_hardi()

    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        viz_backend="fury")
    myafq.viz_bundles()


@pytest.mark.nightly3
def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """
    for n_sessions in [1, 2]:
        n_subjects = 3
        bids_path = create_dummy_bids_path(n_subjects, n_sessions)
        my_afq = api.AFQ(bids_path,
                         dmriprep="synthetic")
        npt.assert_equal(my_afq.data_frame.shape,
                         (n_subjects * n_sessions, 12))


def test_AFQ_custom_bundle_dict():
    bids_path = create_dummy_bids_path(3, 1)
    bundle_dict = api.make_bundle_dict()
    my_afq = api.AFQ(
        bids_path,
        dmriprep="synthetic",
        bundle_info=bundle_dict)


@pytest.mark.nightly2
def test_AFQ_data():
    """
    Test if API can run without prealign
    """
    _, bids_path, _ = get_temp_hardi()

    for use_prealign in [True, False]:
        myafq = api.AFQ(
            bids_path=bids_path,
            dmriprep='vistasoft',
            use_prealign=use_prealign)
        npt.assert_equal(nib.load(myafq.b0[0]).shape,
                         nib.load(myafq['dwi_file'][0]).shape[:3])
        npt.assert_equal(nib.load(myafq.b0[0]).shape,
                         nib.load(myafq.dti[0]).shape[:3])
        myafq.export_rois()


@pytest.mark.nightly5
def test_AFQ_anisotropic():
    """
    Test if API can run using anisotropic registration
    with a specific selection of b vals
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        min_bval=1990,
        max_bval=2010,
        b0_threshold=50,
        reg_template="mni_T1",
        reg_subject="power_map")

    row = myafq.data_frame.iloc[0]
    _, gtab, _ = myafq._get_data_gtab(row)

    # check the b0s mask is correct
    b0s_mask = np.zeros(160, dtype=bool)
    b0s_mask[0:10] = True
    npt.assert_equal(gtab.b0s_mask, b0s_mask)

    # check that only b values in the b val range passed
    bvals_in_range = \
        np.logical_and((gtab.bvals > 1990), (gtab.bvals < 2010))
    bvals_in_range_or_0 = \
        np.logical_or(bvals_in_range, gtab.b0s_mask)
    npt.assert_equal(bvals_in_range_or_0, np.ones(160, dtype=bool))

    # check that the apm map was made
    myafq.export_rois()
    assert op.exists(op.join(
        myafq.data_frame['results_dir'][0],
        'sub-01_ses-01_dwi_model-CSD_APM.nii.gz'))


def test_API_type_checking():
    _, bids_path, _ = get_temp_hardi()
    with pytest.raises(
            TypeError,
            match="bids_path must be a string"):
        api.AFQ(2)

    with pytest.raises(
            TypeError,
            match="custom_tractography_bids_filters must be"
            + " either a dict or None"):
        api.AFQ(
            bids_path,
            custom_tractography_bids_filters=["dwi"])

    with pytest.raises(
            TypeError,
            match="brain_mask must be None or a mask defined in `AFQ.mask`"):
        api.AFQ(
            bids_path,
            brain_mask="not a brain mask")

    with pytest.raises(
            TypeError,
            match="viz_backend must contain either 'fury' or 'plotly'"):
        api.AFQ(bids_path, viz_backend="matplotlib")

    with pytest.raises(
            TypeError,
            match="bundle_info must be None, a list of strings, or a dict"):
        api.AFQ(bids_path, bundle_info=[2, 3])


@pytest.mark.skip(reason="may cause OOM")
def test_AFQ_slr():
    """
    Test if API can run using slr map
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        reg_subject='subject_sls',
        reg_template='hcp_atlas')
    myafq.export_rois()


@pytest.mark.nightly2
def test_AFQ_reco():
    """
    Test if API can run segmentation with recobundles
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        viz_backend="plotly",
        segmentation_params={
            'seg_algo': 'reco',
            'rng': 42})

    tgram = load_tractogram(myafq.get_clean_bundles()[0], myafq.dwi_img[0])
    bundles = aus.tgram_to_bundles(tgram, myafq.bundle_dict, myafq.dwi_img[0])
    npt.assert_(len(bundles['CCMid']) > 0)
    myafq.export_all()


@pytest.mark.nightly4
def test_AFQ_reco80():
    """
    Test API segmentation with the 80-bundle atlas
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        segmentation_params={
            'seg_algo': 'reco80',
            'rng': 42})

    tgram = load_tractogram(myafq.get_clean_bundles()[0], myafq.dwi_img[0])
    bundles = aus.tgram_to_bundles(tgram, myafq.bundle_dict, myafq.dwi_img[0])
    npt.assert_(len(bundles['CCMid']) > 0)


@pytest.mark.nightly2
def test_AFQ_pft():
    """
    Test pft interface for AFQ
    """
    _, bids_path, sub_path = get_temp_hardi()

    bundle_names = ["SLF", "ARC", "CST", "FP"]

    f_pve_csf, f_pve_gm, f_pve_wm = get_fnames('stanford_pve_maps')
    os.rename(f_pve_wm, op.join(sub_path, "sub-01_ses-01_WMprobseg.nii.gz"))
    os.rename(f_pve_gm, op.join(sub_path, "sub-01_ses-01_GMprobseg.nii.gz"))
    os.rename(f_pve_csf, op.join(sub_path, "sub-01_ses-01_CSFprobseg.nii.gz"))

    stop_mask = PFTMask(
        MaskFile("WMprobseg"),
        MaskFile("GMprobseg"),
        MaskFile("CSFprobseg"))

    my_afq = api.AFQ(
        bids_path,
        dmriprep='vistasoft',
        bundle_info=bundle_names,
        tracking_params={
            "stop_mask": stop_mask,
            "stop_threshold": "CMC",
            "tracker": "pft"
        })
    my_afq.export_rois()


@pytest.mark.nightly4
def test_AFQ_custom_subject_reg():
    """
    Test custom subject registration using AFQ object
    """
    # make first temproary directory to generate b0
    _, bids_path, sub_path = get_temp_hardi()

    bundle_names = ["SLF", "ARC", "CST", "FP"]

    b0_file = api.AFQ(
        bids_path,
        dmriprep='vistasoft',
        bundle_info=bundle_names).get_b0()[0]

    # make a different temporary directly to test this custom file in
    _, bids_path, sub_path = get_temp_hardi()

    os.rename(b0_file, op.join(sub_path, "sub-01_ses-01_customb0.nii.gz"))

    my_afq = api.AFQ(
        bids_path,
        dmriprep='vistasoft',
        bundle_info=bundle_names,
        reg_template="mni_T2",
        reg_subject={
            "suffix": "customb0",
            "scope": "vistasoft"})
    my_afq.export_rois()


# Requires large download
@pytest.mark.nightly
def test_AFQ_FA():
    """
    Test if API can run registeration with FA
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        reg_template='dti_fa_template',
        reg_subject='dti_fa_subject')
    myafq.export_rois()


@pytest.mark.nightly
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


def test_auto_cli():
    tmpdir = nbtmp.InTemporaryDirectory()
    config_file = op.join(tmpdir.name, 'test.toml')

    arg_dict = afb.func_dict_to_arg_dict()
    arg_dict['BIDS']['bids_path']['default'] = tmpdir.name
    afb.generate_config(config_file, arg_dict, False)
    with pytest.raises(
            ValueError,
            match="There must be a dataset_description.json in bids_path"):
        afb.parse_config_run_afq(config_file, arg_dict, False)


@pytest.mark.skip(reason="causes segmentation fault")
def test_run_using_auto_cli():
    tmpdir, bids_path, _ = get_temp_hardi()
    config_file = op.join(tmpdir.name, 'test.toml')

    arg_dict = afb.func_dict_to_arg_dict()

    # set our custom defaults for the toml file
    # It is easier to edit them here, than to parse the file and edit them
    # after the file is written
    arg_dict['BIDS']['bids_path']['default'] = bids_path
    arg_dict['BIDS']['dmriprep']['default'] = 'vistasoft'
    arg_dict['BUNDLES']['bundle_names']['default'] = ["CST"]
    arg_dict['TRACTOGRAPHY']['n_seeds']['default'] = 500
    arg_dict['TRACTOGRAPHY']['random_seeds']['default'] = True

    afb.generate_config(config_file, arg_dict, False)
    afb.parse_config_run_afq(config_file, arg_dict, False)


def test_AFQ_data_waypoint():
    """
    Test with some actual data again, this time for track segmentation
    """
    tmpdir, bids_path, _ = get_temp_hardi()
    bundle_names = ["SLF", "ARC", "CST", "FP"]
    tracking_params = dict(odf_model="dti",
                           seed_mask=RoiMask(),
                           n_seeds=100,
                           random_seeds=True,
                           rng_seed=42)
    segmentation_params = dict(filter_by_endpoints=False,
                               seg_algo="AFQ",
                               return_idx=True)

    clean_params = dict(return_idx=True)

    myafq = api.AFQ(bids_path=bids_path,
                    dmriprep='vistasoft',
                    bundle_info=bundle_names,
                    scalars=["dti_FA", "dti_MD"],
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params,
                    clean_params=clean_params)

    # Replace the mapping and streamlines with precomputed:
    file_dict = afd.read_stanford_hardi_tractography()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    streamlines = dts.Streamlines(
        dtu.transform_tracking_output(
            [s for s in streamlines if s.shape[0] > 100],
            np.linalg.inv(myafq.dwi_affine[0])))

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
    npt.assert_(len(bundles['CST_L']) > 0)

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
    assert tract_profiles.shape == (500, 5)

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

    # Set up config to use the same parameters as above:
    # ROI mask needs to be put in quotes in config
    tracking_params = dict(odf_model="DTI",
                           seed_mask="RoiMask()",
                           n_seeds=100,
                           random_seeds=True,
                           rng_seed=42)
    config = dict(BIDS=dict(bids_path=bids_path,
                            dmriprep='vistasoft'),
                  BUNDLES=dict(
                      bundle_info=bundle_names,
                      scalars=["dti_fa", "dti_md"]),
                  VIZ=dict(
                      viz_backend="plotly_no_gif"),
                  TRACTOGRAPHY=tracking_params,
                  SEGMENTATION=segmentation_params,
                  CLEANING=clean_params)

    config_file = op.join(tmpdir.name, "afq_config.toml")
    with open(config_file, 'w') as ff:
        toml.dump(config, ff)

    cmd = "pyAFQ " + config_file
    out = os.system(cmd)
    assert out == 0
    # The combined tract profiles should already exist from the CLI Run:
    from_file = pd.read_csv(
        myafq._get_fname(myafq.data_frame.iloc[0], '_profiles.csv'))
    # And should be identical to what we would get by rerunning this:
    combined_profiles = myafq.combine_profiles()
    assert combined_profiles.shape == (500, 7)
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


@pytest.mark.nightly3
def test_afq_msmt():
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_cfin_data(path=tmpdir.name)
    myafq = api.AFQ(bids_path=op.join(tmpdir.name, 'cfin_multib'),
                    dmriprep='dipy', tracking_params={"odf_model": "MSMT"})
    npt.assert_equal(
        op.split(myafq.streamlines[0])[-1],
        "sub-01_ses-01_dwi_space-RASMM_model-MSMT_desc-det_tractography.trk")
