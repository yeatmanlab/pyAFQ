import tempfile
import os
import os.path as op
import shutil
import subprocess
import gc

import toml

import numpy as np
import numpy.testing as npt
import pytest

import pandas as pd
from pandas.testing import assert_series_equal

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
import AFQ.utils.bin as afb
from AFQ.definitions.mask import RoiMask, ThresholdedScalarMask,\
    PFTMask, MaskFile
from AFQ.definitions.mapping import SynMap, AffMap, SlrMap
from AFQ.definitions.scalar import TemplateScalar


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


def create_dummy_data(dmriprep_dir, subject, session=None):
    aff = np.eye(4)
    data = np.ones((10, 10, 10, 6))
    bvecs = np.vstack([np.eye(3), np.eye(3)])
    bvecs[0] = 0
    bvals = np.ones(6) * 1000.
    bvals[0] = 0

    if session is None:
        data_dir = subject
    else:
        data_dir = op.join(subject, session)

    np.savetxt(
        op.join(
            dmriprep_dir, data_dir, 'dwi', 'dwi.bval'),
        bvals)
    np.savetxt(
        op.join(
            dmriprep_dir, data_dir, 'dwi', 'dwi.bvec'),
        bvecs)
    nib.save(
        nib.Nifti1Image(data, aff),
        op.join(
            dmriprep_dir, data_dir, 'dwi', 'dwi.nii.gz'))
    nib.save(
        nib.Nifti1Image(data, aff),
        op.join(
            dmriprep_dir, data_dir, 'anat', 'T1w.nii.gz'))
    nib.save(
        nib.Nifti1Image(data, aff),
        op.join(
            dmriprep_dir, data_dir, 'anat', 'seg.nii.gz'))


def create_dummy_bids_path(n_subjects, n_sessions, share_sessions=True):
    subjects = ['sub-0%s' % (d + 1) for d in range(n_subjects)]

    # Case where there are individual session folders within each subject's
    # folder:
    if n_sessions > 1:
        # create data for n_sessions for each subject
        if share_sessions:
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
                    create_dummy_data(dmriprep_dir, subject, session)
        else:
            # create different sessions for each subject
            sessions = ['ses-0%s' % (d + 1) for d in range(n_subjects)]

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

            for d in range(n_subjects):
                subject = subjects[d]
                session = sessions[d]
                for modality in ['anat', 'dwi']:
                    os.makedirs(
                        op.join(dmriprep_dir, subject, session, modality))
                # Make some dummy data:
                create_dummy_data(dmriprep_dir, subject, session)
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
            create_dummy_data(dmriprep_dir, subject)

    return bids_dir


def test_BundleDict():
    """
    Tests bundle dict
    """

    # test defaults
    afq_bundles = api.BundleDict()

    # bundles restricted within hemisphere
    # NOTE: FA and FP cross midline so are removed
    # NOTE: all others generate two bundles
    num_hemi_bundles = (len(api.BUNDLES)-2)*2

    # bundles that cross the midline
    num_whole_bundles = 2

    assert len(afq_bundles) == num_hemi_bundles + num_whole_bundles

    # Arcuate Fasciculus
    afq_bundles = api.BundleDict(["ARC"])

    assert len(afq_bundles) == 2

    # Forceps Minor
    afq_bundles = api.BundleDict(["FA"])

    assert len(afq_bundles) == 1

    # Cingulum Hippocampus
    # not included but exists in templates
    afq_bundles = api.BundleDict(["HCC"])

    assert len(afq_bundles) == 2

    # Test "custom" bundle
    afq_templates = afd.read_templates()
    afq_bundles = api.BundleDict({
        "custom_bundle": {
            "ROIs": [afq_templates["FA_L"],
                     afq_templates["FP_R"]],
            "rules": [True, True],
            "cross_midline": False,
            "uid": 1}})
    afq_bundles.get("custom_bundle")

    assert len(afq_bundles) == 1

    # Vertical Occipital Fasciculus
    # not included and does not exist in afq templates
    with pytest.raises(
            ValueError,
            match="VOF_R is not in AFQ templates"):
        afq_bundles = api.BundleDict(["VOF"])
        afq_bundles["VOF_R"]

    afq_bundles = api.BundleDict(["VOF"], seg_algo="reco80")
    assert len(afq_bundles) == 2

    afq_bundles = api.BundleDict(["whole_brain"], seg_algo="reco80")
    assert len(afq_bundles) == 1


def test_AFQ_missing_files():
    tmpdir = nbtmp.InTemporaryDirectory()
    bids_path = tmpdir.name

    with pytest.raises(
            ValueError,
            match="There must be a dataset_description.json in bids_path"):
        api.AFQ(bids_path)
    afd.to_bids_description(
        bids_path,
        **{"Name": "Missing", "Subjects": ["sub-01"]})

    with pytest.raises(
            ValueError,
            match=f"No non-json files recognized by pyBIDS in {bids_path}"):
        api.AFQ(bids_path)

    subses_folder = op.join(
        bids_path,
        "derivatives",
        "otherDeriv",
        'sub-01',
        'ses-01')
    os.makedirs(subses_folder, exist_ok=True)
    afd.to_bids_description(
        op.join(
            bids_path,
            "derivatives",
            "otherDeriv"),
        **{
            "Name": "Missing",
            "PipelineDescription": {"Name": "otherDeriv"}})
    touch(op.join(subses_folder, "sub-01_ses-01_dwi.nii.gz"))

    with pytest.raises(
            ValueError,
            match="No non-json files recognized by pyBIDS"
            + " in the pipeline: missingPipe"):
        api.AFQ(bids_path, dmriprep="missingPipe")

    os.mkdir(op.join(bids_path, "missingPipe"))
    afd.to_bids_description(
        op.join(bids_path, "missingPipe"), **{
            "Name": "Missing",
            "PipelineDescription": {"Name": "missingPipe"}})
    with pytest.raises(
            ValueError,
            match="No non-json files recognized by pyBIDS"
            + " in the pipeline: missingPipe"):
        api.AFQ(bids_path, dmriprep="missingPipe")


@pytest.mark.nightly_custom
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
    my_afq.export_streamlines()


@pytest.mark.nightly_basic
def test_AFQ_no_derivs():
    """
    Test the initialization of the AFQ object
    """
    bids_path = create_dummy_bids_path(1, 1)
    os.remove(op.join(
        bids_path, "derivatives", "dmriprep", "dataset_description.json"))
    with pytest.raises(
            ValueError,
            match=f"No non-json files recognized by pyBIDS in {bids_path}"):
        api.AFQ(
            bids_path,
            dmriprep="synthetic")


@pytest.mark.nightly_custom
@xvfb_it
def test_AFQ_fury():
    _, bids_path, _ = get_temp_hardi()

    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        viz_backend="fury")
    myafq.all_bundles_figure


@pytest.mark.nightly_msmt_and_init
def test_AFQ_init():
    """
    Test the initialization of the AFQ object
    """
    for n_sessions in [1, 2, 3]:
        n_subjects = 3
        bids_path = create_dummy_bids_path(n_subjects, n_sessions,
                                           (n_subjects != n_sessions))
        my_afq = api.AFQ(bids_path,
                         dmriprep="synthetic")

        for subject in range(n_subjects):
            sub = f"0{subject+1}"
            if n_subjects == n_sessions:
                npt.assert_equal(
                    len(my_afq.wf_dict[sub][sub]),
                    40)
            else:
                for session in range(n_sessions):
                    if n_sessions == 1:
                        sess = "None"
                    else:
                        sess = f"0{session+1}"
                    npt.assert_equal(
                        len(my_afq.wf_dict[sub][sess]),
                        40)


def test_AFQ_custom_bundle_dict():
    bids_path = create_dummy_bids_path(3, 1)
    bundle_dict = api.BundleDict()
    api.AFQ(
        bids_path,
        dmriprep="synthetic",
        bundle_info=bundle_dict)


@pytest.mark.nightly_basic
def test_AFQ_data():
    """
    Test if API can run without prealign and with only pre-align
    """
    _, bids_path, _ = get_temp_hardi()

    for mapping in [SynMap(use_prealign=False), AffMap()]:
        myafq = api.AFQ(
            bids_path=bids_path,
            dmriprep='vistasoft',
            mapping=mapping)
        npt.assert_equal(nib.load(myafq.b0["01"]).shape,
                         nib.load(myafq.dwi_file["01"]).shape[:3])
        npt.assert_equal(nib.load(myafq.b0["01"]).shape,
                         nib.load(myafq.dti_params["01"]).shape[:3])
        myafq.rois
        shutil.rmtree(op.join(
            bids_path,
            'derivatives/afq'))


@pytest.mark.nightly_anisotropic
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

    gtab = myafq.gtab["01"]

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
    myafq.mapping
    assert op.exists(op.join(
        myafq.results_dir["01"],
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
            match=("brain_mask must be None or a mask defined"
                   " in `AFQ.definitions.mask`")):
        api.AFQ(
            bids_path,
            brain_mask="not a brain mask")

    with pytest.raises(
            TypeError,
            match="viz_backend must contain either 'fury' or 'plotly'"):
        api.AFQ(bids_path, viz_backend="matplotlib")

    with pytest.raises(
            TypeError,
            match=(
                "bundle_info must be None, a list of strings,"
                " a dict, or a BundleDict")):
        api.AFQ(bids_path, bundle_info=[2, 3])


@pytest.mark.nightly_slr
def test_AFQ_slr():
    """
    Test if API can run using slr map
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        reg_subject='subject_sls',
        reg_template='hcp_atlas',
        mapping=SlrMap())

    tgram = load_tractogram(myafq.clean_bundles["01"], myafq.img["01"])
    bundles = aus.tgram_to_bundles(
        tgram, myafq.bundle_dict, myafq.img["01"])
    npt.assert_(len(bundles['CST_L']) > 0)


@pytest.mark.nightly_reco
def test_AFQ_reco():
    """
    Test if API can run segmentation with recobundles
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = api.AFQ(
        bids_path=bids_path,
        dmriprep='vistasoft',
        viz_backend="plotly",
        profile_weights="median",
        segmentation_params={
            'seg_algo': 'reco',
            'rng': 42})

    tgram = load_tractogram(myafq.clean_bundles["01"], myafq.img["01"])
    bundles = aus.tgram_to_bundles(tgram, myafq.bundle_dict, myafq.img["01"])
    npt.assert_(len(bundles['CCMid']) > 0)
    myafq.export_all()


@pytest.mark.nightly_custom
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

    tgram = load_tractogram(myafq.clean_bundles["01"], myafq.img["01"])
    bundles = aus.tgram_to_bundles(tgram, myafq.bundle_dict, myafq.img["01"])
    npt.assert_(len(bundles['CCMid']) > 0)


@pytest.mark.nightly_pft
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
    my_afq.export_streamlines()


@pytest.mark.nightly_custom
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
        bundle_info=bundle_names).b0["01"]

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
    myafq.rois


@pytest.mark.nightly
def test_DKI_profile():
    """
    Test using API to profile dki
    """
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_cfin_data(path=tmpdir.name)
    myafq = api.AFQ(bids_path=op.join(tmpdir.name, 'cfin_multib'),
                    dmriprep='dipy')
    myafq.dki_fa
    myafq.dki_md


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
    t1_path = op.join(tmpdir.name, "T1.nii.gz")
    nib.save(
        afd.read_mni_template(mask=True, weight="T1w"),
        t1_path)

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
                    scalars=[
                        "dti_FA",
                        "dti_MD",
                        TemplateScalar("T1", t1_path)],
                    robust_tensor_fitting=True,
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
            np.linalg.inv(myafq.dwi_affine["01"])))

    mapping_file = op.join(
        myafq.results_dir["01"],
        'sub-01_ses-01_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz')
    nib.save(mapping, mapping_file)
    reg_prealign_file = op.join(
        myafq.results_dir["01"],
        'sub-01_ses-01_dwi_prealign_from-DWI_to-MNI_xfm.npy')
    np.save(reg_prealign_file, np.eye(4))

    tgram = load_tractogram(myafq.bundles["01"], myafq.img["01"])

    bundles = aus.tgram_to_bundles(
        tgram, myafq.bundle_dict, myafq.img["01"])
    npt.assert_(len(bundles['CST_L']) > 0)

    # Test ROI exporting:
    myafq.export_rois()
    assert op.exists(op.join(
        myafq.results_dir["01"],
        'ROIs',
        'sub-01_ses-01_dwi_desc-ROI-CST_R-1-include.json'))

    # Test bundles exporting:
    myafq.export_indiv_bundles()
    assert op.exists(op.join(
        myafq.results_dir["01"],
        'bundles',
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ-CST_L_tractography.trk'))  # noqa

    tract_profile_fname = myafq.profiles["01"]
    tract_profiles = pd.read_csv(tract_profile_fname)
    assert tract_profiles.shape == (500, 6)

    myafq.tract_profile_plots
    assert op.exists(op.join(
        myafq.results_dir["01"],
        "tract_profile_plots",
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ_dti_fa_profile_plots.png'))  # noqa

    assert op.exists(op.join(
        myafq.results_dir["01"],
        "tract_profile_plots",
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ_dti_md_profile_plots.png'))  # noqa

    # Before we run the CLI, we'll remove the bundles and ROI folders, to see
    # that the CLI generates them
    shutil.rmtree(op.join(myafq.results_dir["01"],
                          'bundles'))

    shutil.rmtree(op.join(myafq.results_dir["01"],
                          'ROIs'))
    os.remove(tract_profile_fname)

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
                  DATA=dict(
                      robust_tensor_fitting=True),
                  BUNDLES=dict(
                      bundle_info=bundle_names,
                      scalars=[
                        "dti_fa",
                        "dti_md",
                        f"TemplateScalar('T1', '{t1_path}')"]),
                  VIZ=dict(
                      viz_backend="plotly_no_gif"),
                  TRACTOGRAPHY=tracking_params,
                  SEGMENTATION=segmentation_params,
                  CLEANING=clean_params)

    config_file = op.join(tmpdir.name, "afq_config.toml")
    with open(config_file, 'w') as ff:
        toml.dump(config, ff)

    # save memory
    results_dir = myafq.results_dir["01"]
    del myafq
    gc.collect()

    cmd = "pyAFQ " + config_file
    completed_process = subprocess.run(
        cmd, shell=True, capture_output=True)
    if completed_process.returncode != 0:
        print(completed_process.stdout)
    print(completed_process.stderr)
    assert completed_process.returncode == 0
    # The tract profiles should already exist from the CLI Run:
    from_file = pd.read_csv(tract_profile_fname)

    assert from_file.shape == (500, 6)
    assert_series_equal(tract_profiles['dti_fa'], from_file['dti_fa'])

    # Make sure the CLI did indeed generate these:
    assert op.exists(op.join(
        results_dir,
        'ROIs',
        'sub-01_ses-01_dwi_desc-ROI-CST_R-1-include.json'))

    assert op.exists(op.join(
        results_dir,
        'bundles',
        'sub-01_ses-01_dwi_space-RASMM_model-DTI_desc-det-AFQ-CST_L_tractography.trk'))  # noqa


@pytest.mark.nightly_msmt_and_init
def test_afq_msmt():
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_cfin_data(path=tmpdir.name)
    myafq = api.AFQ(bids_path=op.join(tmpdir.name, 'cfin_multib'),
                    dmriprep='dipy', tracking_params={"odf_model": "MSMT"})
    npt.assert_equal(
        op.split(myafq.streamlines["01"])[-1],
        "sub-01_ses-01_dwi_space-RASMM_model-MSMT_desc-det_tractography.trk")
