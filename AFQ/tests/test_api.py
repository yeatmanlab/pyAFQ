import tempfile
import os
import os.path as op
import shutil
import subprocess
import gc
import random

import toml

import numpy as np
import numpy.testing as npt
import pytest

import pandas as pd
from pandas.testing import assert_series_equal

import nibabel as nib
import tempfile

import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
from dipy.data import get_fnames
from dipy.testing.decorators import xvfb_it
from dipy.io.streamline import load_tractogram

from AFQ.api.bundle_dict import BundleDict
from AFQ.api.group import GroupAFQ
from AFQ.api.participant import ParticipantAFQ
import AFQ.data.fetch as afd
import AFQ.utils.streamlines as aus
import AFQ.utils.bin as afb
from AFQ.definitions.mapping import SynMap, AffMap, SlrMap, IdentityMap
from AFQ.definitions.image import (RoiImage, PFTImage, ImageFile,
                                   TemplateImage, LabelledImageFile)


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def get_temp_hardi():
    tmpdir = tempfile.mkdtemp()
    bids_path = op.join(tmpdir, "stanford_hardi")
    afd.organize_stanford_data(path=tmpdir)

    sub_path = op.join(
        bids_path,
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


def test_AFQ_missing_files():
    tmpdir = tempfile.TemporaryDirectory()
    bids_path = tmpdir.name

    with pytest.raises(
            ValueError,
            match="There must be a dataset_description.json in bids_path"):
        GroupAFQ(bids_path)
    afd.to_bids_description(
        bids_path,
        **{"Name": "Missing", "Subjects": ["sub-01"]})

    with pytest.raises(
            ValueError,
            match=f"No non-json files recognized by pyBIDS in {bids_path}"):
        GroupAFQ(bids_path)

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
        GroupAFQ(bids_path, preproc_pipeline="missingPipe")

    os.mkdir(op.join(bids_path, "missingPipe"))
    afd.to_bids_description(
        op.join(bids_path, "missingPipe"), **{
            "Name": "Missing",
            "PipelineDescription": {"Name": "missingPipe"}})
    with pytest.raises(
            ValueError,
            match="No non-json files recognized by pyBIDS"
            + " in the pipeline: missingPipe"):
        GroupAFQ(bids_path, preproc_pipeline="missingPipe")


@pytest.mark.nightly_custom
def test_AFQ_custom_tract():
    """
    Test whether AFQ can use tractography from
    import_tract
    """
    _, bids_path, sub_path = get_temp_hardi()
    afd.fetch_stanford_hardi_tractography()

    bundle_names = [
        "SLF_L", "SLF_R", "ARC_L", "ARC_R", "CST_L", "CST_R", "FP"]

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
    myafq = GroupAFQ(
        bids_path,
        preproc_pipeline='vistasoft',
        bundle_info=bundle_names,
        import_tract={
            "suffix": "tractography",
            "scope": "vistasoft"
        })

    # equivalent ParticipantAFQ version of call may be useful as reference
    # myafq = ParticipantAFQ(
    #     sub_path + "/sub-01_ses-01_dwi.nii.gz",
    #     sub_path + "/sub-01_ses-01_dwi.bval",
    #     sub_path + "/sub-01_ses-01_dwi.bvec",
    #     output_dir=sub_path,
    #     bundle_info=bundle_names,
    #     import_tract=sub_path + '/subsampled_tractography.trk')

    myafq.export("streamlines")


@pytest.mark.nightly_basic
def test_AFQ_no_derivs():
    """
    Test the initialization of the GroupAFQ object
    """
    bids_path = create_dummy_bids_path(1, 1)
    os.remove(op.join(
        bids_path, "derivatives", "dmriprep", "dataset_description.json"))
    with pytest.raises(
            ValueError,
            match=f"No non-json files recognized by pyBIDS in {bids_path}"):
        GroupAFQ(
            bids_path,
            preproc_pipeline="synthetic")


@pytest.mark.nightly_custom
@xvfb_it
def test_AFQ_fury():
    tmpdir = tempfile.TemporaryDirectory()
    bids_path = op.join(tmpdir.name, "stanford_hardi")
    afd.organize_stanford_data(path=tmpdir.name)

    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        viz_backend_spec="fury")
    myafq.export("all_bundles_figure")

def test_AFQ_trx():
    tmpdir = tempfile.TemporaryDirectory()
    bids_path = op.join(tmpdir.name, "stanford_hardi")
    afd.organize_stanford_data(path=tmpdir.name)

    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        tracking_params={"trx": True})
    myafq.export("all_bundles_figure")


def test_AFQ_init():
    """
    Test the initialization of the GroupAFQ object
    """
    for n_sessions in [1, 2, 3]:
        if n_sessions == 3:
            # we only need to test all of these once
            participant_labels_to_test = [None, ["01"], ["04"]]
        else:
            participant_labels_to_test = [None]
        for participant_labels in participant_labels_to_test:
            if participant_labels is None:
                n_subjects = 3
            else:
                n_subjects = 1
            bids_path = create_dummy_bids_path(
                n_subjects, n_sessions,
                (n_subjects != n_sessions))

            if participant_labels is not None and\
                    participant_labels[0] == "04":
                with pytest.raises(
                    ValueError,
                    match="No subjects specified in `participant_labels` "
                    + " found in BIDS derivatives folders."
                        + " See above warnings."):
                    myafq = GroupAFQ(
                        bids_path,
                        preproc_pipeline="synthetic",
                        participant_labels=participant_labels)
            else:
                myafq = GroupAFQ(
                    bids_path,
                    preproc_pipeline="synthetic",
                    participant_labels=participant_labels)

                for subject in range(n_subjects):
                    sub = f"0{subject+1}"
                    if n_subjects == n_sessions:
                        npt.assert_equal(
                            len(myafq.wf_dict[sub][sub]),
                            28)
                    else:
                        for session in range(n_sessions):
                            if n_sessions == 1:
                                sess = "None"
                            else:
                                sess = f"0{session+1}"
                            npt.assert_equal(
                                len(myafq.wf_dict[sub][sess]),
                                28)


@pytest.mark.nightly_basic
def test_AFQ_data():
    """
    Test if API can run without prealign and with only pre-align
    """
    _, bids_path, _ = get_temp_hardi()

    for mapping in [SynMap(use_prealign=False), AffMap()]:
        myafq = GroupAFQ(
            bids_path=bids_path,
            preproc_pipeline='vistasoft',
            mapping_definition=mapping)
        npt.assert_equal(nib.load(myafq.export("b0")["01"]).shape,
                         nib.load(myafq.export("dwi")["01"]).shape[:3])
        npt.assert_equal(nib.load(myafq.export("b0")["01"]).shape,
                         nib.load(myafq.export("dti_params")["01"]).shape[:3])
        myafq.export("rois")
        myafq.cmd_outputs()
        assert len(os.listdir(op.join(
            myafq.afq_path, "sub-01", "ses-01"))) == 0


@pytest.mark.nightly_anisotropic
def test_AFQ_anisotropic():
    """
    Test if API can run using anisotropic power map registration
    with a specific selection of b vals
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        min_bval=1990,
        max_bval=2010,
        b0_threshold=50,
        reg_template_spec="mni_T1",
        reg_subject_spec="power_map")

    gtab = myafq.export("gtab")["01"]

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
    myafq.export("mapping")
    assert op.exists(op.join(
        myafq.export("results_dir")["01"],
        'sub-01_ses-01_dwi_model-CSD_desc-APM_dwi.nii.gz'))


def test_API_type_checking():
    _, bids_path, _ = get_temp_hardi()
    np.random.seed(2022)
    # Note that the ordering of these "with pytest.raises"
    # statements is important. Later tests will use
    # the sucessful results of previous ones. IE,
    # do not put a test of brain masks in front of a test
    # of all_bundles_figure, because it will use the brain mask
    # generated by the all_bundles_figure test isntead of testing
    # a new brain mask.
    with pytest.raises(
            TypeError,
            match="bids_path must be a string"):
        GroupAFQ(2)

    with pytest.raises(
            TypeError,
            match=(
                "import_tract must be"
                " either a dict or a str")):
        myafq = GroupAFQ(
            bids_path,
            preproc_pipeline='vistasoft',
            import_tract=["dwi"])
        myafq.export("streamlines")
    del myafq

    with pytest.raises(
            TypeError,
            match="brain_mask_definition must be a Definition"):
        myafq = GroupAFQ(
            bids_path,
            preproc_pipeline='vistasoft',
            brain_mask_definition="not a brain mask")

    with pytest.raises(
            ValueError,
            match=r"No file found with these parameters:\n*"):
        myafq = GroupAFQ(
            bids_path,
            preproc_pipeline='vistasoft',
            brain_mask_definition=ImageFile(
                suffix='dne_dne',
                filters={'scope': 'dne_dne'}))
        myafq.export("brain_mask")

    with pytest.raises(
            TypeError,
            match=(
                "bundle_info must be a list of strings,"
                " a dict, or a BundleDict")):
        myafq = GroupAFQ(
            bids_path,
            preproc_pipeline='vistasoft',
            bundle_info=[2, 3])
        myafq.export("bundle_dict")
    del myafq

    with pytest.raises(
            ValueError,
            match=(
                "Fatal: No bundles recognized.")):
        myafq = GroupAFQ(
            bids_path,
            preproc_pipeline='vistasoft',
            mapping_definition=IdentityMap(),
            tracking_params={
                "n_seeds": 1,
                "random_seeds": True,
                "directions": "prob",
                "odf_model": "CSD"},
            csd_sh_order=2,  # reduce CSD fit time
            csd_response=((1, 1, 1), 1),
            bundle_info=["ARC_L", "ARC_R"])
        myafq.export("bundles")
    del myafq

    with pytest.raises(
            TypeError,
            match="viz_backend_spec must contain either 'fury' or 'plotly'"):
        myafq = GroupAFQ(
            bids_path,
            preproc_pipeline='vistasoft',
            viz_backend_spec="matplotlib")
        myafq.export("viz_backend")
    del myafq


def test_AFQ_slr():
    """
    Test if API can run using slr map
    """
    seed = 2022
    random.seed(seed)

    afd.read_stanford_hardi_tractography()

    _, bids_path, _ = get_temp_hardi()
    bd = BundleDict(["CST_L"])

    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        reg_subject_spec='subject_sls',
        reg_template_spec='hcp_atlas',
        import_tract=op.join(
            op.expanduser('~'),
            'AFQ_data',
            'stanford_hardi_tractography',
            'full_segmented_cleaned_tractography.trk'),
        segmentation_params={
            "dist_to_waypoint": 10,
            "filter_by_endpoints": False,
            "parallel_segmentation": {"engine": "serial"}},
        bundle_info=bd,
        mapping_definition=SlrMap(slr_kwargs={
            "rng": np.random.RandomState(seed)}))

    seg_sft = aus.SegmentedSFT.fromfile(
        myafq.export("bundles")["01"])
    npt.assert_(len(seg_sft.get_bundle('CST_L').streamlines) > 0)


@pytest.mark.nightly_reco
def test_AFQ_reco():
    """
    Test if API can run segmentation with recobundles
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        viz_backend_spec="plotly",
        profile_weights="median",
        segmentation_params={
            'seg_algo': 'reco',
            'rng': 42})

    seg_sft = aus.SegmentedSFT.fromfile(
        myafq.export("bundles")["01"])
    npt.assert_(len(seg_sft.get_bundle('CCMid').streamlines) > 0)
    myafq.export_all()


@pytest.mark.nightly_reco80
def test_AFQ_reco80():
    """
    Test API segmentation with the 80-bundle atlas
    """
    _, bids_path, _ = get_temp_hardi()

    tracking_params = dict(
        odf_model="csd",
        n_seeds=10000,
        random_seeds=True,
        rng_seed=42)

    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        tracking_params=tracking_params,
        segmentation_params={
            'seg_algo': 'reco80',
            'rng': 42})

    seg_sft = aus.SegmentedSFT.fromfile(
        myafq.export("bundles")["01"])
    npt.assert_(len(seg_sft.get_bundle('CCMid').streamlines) > 0)


@pytest.mark.nightly_pft
def test_AFQ_pft():
    """
    Test pft interface for AFQ
    """
    _, bids_path, sub_path = get_temp_hardi()

    bundle_names = [
        "SLF_L", "SLF_R", "ARC_L", "ARC_R", "CST_L", "CST_R", "FP"]

    f_pve_csf, f_pve_gm, f_pve_wm = get_fnames('stanford_pve_maps')
    os.rename(f_pve_wm, op.join(sub_path, "sub-01_ses-01_WMprobseg.nii.gz"))
    os.rename(f_pve_gm, op.join(sub_path, "sub-01_ses-01_GMprobseg.nii.gz"))
    os.rename(f_pve_csf, op.join(sub_path, "sub-01_ses-01_CSFprobseg.nii.gz"))

    stop_mask = PFTImage(
        ImageFile(suffix="WMprobseg"),
        ImageFile(suffix="GMprobseg"),
        ImageFile(suffix="CSFprobseg"))
    t_output_dir = tempfile.TemporaryDirectory()

    myafq = GroupAFQ(
        bids_path,
        preproc_pipeline='vistasoft',
        bundle_info=bundle_names,
        output_dir=t_output_dir.name,
        tracking_params={
            "stop_mask": stop_mask,
            "stop_threshold": "CMC",
            "tracker": "pft",
            "max_length": 150,
        })
    sl_file = myafq.export("streamlines")["01"]
    dwi_file = myafq.export("dwi")["01"]
    sls = load_tractogram(
        sl_file,
        dwi_file,
        bbox_valid_check=False,
        trk_header_check=False).streamlines
    for sl in sls:
        # double the max_length, due to step size of 0.5
        assert len(sl) <= 300


@pytest.mark.nightly_custom
def test_AFQ_custom_subject_reg():
    """
    Test custom subject registration using GroupAFQ object
    """
    # make first temproary directory to generate b0
    _, bids_path, sub_path = get_temp_hardi()

    bundle_info = BundleDict([
        "SLF_L", "SLF_R", "ARC_L", "ARC_R", "CST_L", "CST_R", "FP"])

    b0_file = GroupAFQ(
        bids_path,
        preproc_pipeline='vistasoft',
        bundle_info=bundle_info).export("b0")["01"]

    # make a different temporary directly to test this custom file in
    _, bids_path, sub_path = get_temp_hardi()

    os.rename(b0_file, op.join(sub_path, "sub-01_ses-01_customb0.nii.gz"))

    myafq = GroupAFQ(
        bids_path,
        preproc_pipeline='vistasoft',
        bundle_info=bundle_info,
        reg_template_spec="mni_T2",
        reg_subject_spec=ImageFile(
            suffix="customb0",
            filters={"scope": "vistasoft"}))
    myafq.export("rois")


# Requires large download
@pytest.mark.nightly
def test_AFQ_FA():
    """
    Test if API can run registeration with FA
    """
    _, bids_path, _ = get_temp_hardi()
    myafq = GroupAFQ(
        bids_path=bids_path,
        preproc_pipeline='vistasoft',
        reg_template_spec='dti_fa_template',
        reg_subject_spec='dti_fa_subject')
    myafq.export("rois")


@pytest.mark.nightly
def test_multib_profile():
    """
    Test using API to profile DKI/fwDTI
    """
    tmpdir = tempfile.TemporaryDirectory()
    afd.organize_cfin_data(path=tmpdir.name)
    myafq = GroupAFQ(bids_path=op.join(tmpdir.name, 'cfin_multib'),
                    preproc_pipeline='dipy')
    myafq.export("dki_fa")
    myafq.export("dki_md")
    myafq.export("fwdti_fa")
    myafq.export("fwdti_md")


def test_auto_cli():
    tmpdir = tempfile.TemporaryDirectory()
    config_file = op.join(tmpdir.name, 'test.toml')

    arg_dict = afb.func_dict_to_arg_dict()
    arg_dict['BIDS_PARAMS']['bids_path']['default'] = tmpdir.name
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
    arg_dict['BIDS_PARAMS']['bids_path']['default'] = bids_path
    arg_dict['BIDS_PARAMS']['dmriprep']['default'] = 'vistasoft'
    arg_dict['DATA']['bundle_info']['default'] = ["CST_L"]
    arg_dict['TRACTOGRAPHY_PARAMS']['n_seeds']['default'] = 500
    arg_dict['TRACTOGRAPHY_PARAMS']['random_seeds']['default'] = True

    afb.generate_config(config_file, arg_dict, False)
    afb.parse_config_run_afq(config_file, arg_dict, False)


def test_AFQ_data_waypoint():
    """
    Test with some actual data again, this time for track segmentation
    """
    tmpdir, bids_path, _ = get_temp_hardi()
    t1_path = op.join(tmpdir, "T1.nii.gz")
    t1_path_other = op.join(tmpdir, "T1-untransformed.nii.gz")
    nib.save(
        afd.read_mni_template(mask=True, weight="T1w"),
        t1_path)
    shutil.copy(t1_path, t1_path_other)

    vista_folder = op.join(
        bids_path,
        "derivatives/vistasoft/sub-01/ses-01/dwi")

    # Prepare LV1 ROI
    lv1_files, lv1_folder = afd.fetch_stanford_hardi_lv1()
    lv1_fname = op.join(
        lv1_folder,
        list(lv1_files.keys())[0])
    bundle_names = [
        "SLF_L", "SLF_R", "ARC_L", "ARC_R", "CST_L", "CST_R", "FP"]
    bundle_info = BundleDict(
        bundle_names,
        resample_subject_to=nib.load(
            op.join(vista_folder, "sub-01_ses-01_dwi.nii.gz")))

    # Test when we have endpoint ROIs only
    bundle_info.load_templates()
    bundle_info["SLF_L"] = {
        "start": bundle_info.templates["SLF_L_start"],
        "end": bundle_info.templates["SLF_L_end"],
        "prob_map" : bundle_info.templates["SLF_L_prob_map"]
    }

    bundle_info["LV1"] = {
        "start": lv1_fname,
        "space": "subject"
    }

    tracking_params = dict(odf_model="csd",
                           seed_mask=RoiImage(),
                           n_seeds=100,
                           random_seeds=True,
                           rng_seed=42)
    segmentation_params = dict(filter_by_endpoints=False,
                               seg_algo="AFQ",
                               return_idx=True)

    clean_params = dict(return_idx=True)

    afq_folder = op.join(bids_path, "derivatives/afq/sub-01/ses-01")
    os.makedirs(afq_folder, exist_ok=True)
    myafq = ParticipantAFQ(
        op.join(vista_folder, "sub-01_ses-01_dwi.nii.gz"),
        op.join(vista_folder, "sub-01_ses-01_dwi.bval"),
        op.join(vista_folder, "sub-01_ses-01_dwi.bvec"),
        afq_folder,
        bundle_info=bundle_info,
        scalars=[
            "dti_FA",
            "dti_MD",
            "dti_GA",
            "dti_lt2",
            ImageFile(path=t1_path_other),
            TemplateImage(t1_path)],
        tracking_params=tracking_params,
        segmentation_params=segmentation_params,
        clean_params=clean_params)

    # Replace the mapping and streamlines with precomputed:
    file_dict = afd.read_stanford_hardi_tractography()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    dwi_affine = myafq.export("dwi_affine")
    streamlines = dts.Streamlines(
        dtu.transform_tracking_output(
            [s for s in streamlines if s.shape[0] > 100],
            np.linalg.inv(dwi_affine)))

    mapping_file = op.join(
        myafq.export("results_dir"),
        'sub-01_ses-01_dwi_desc-mapping_from-DWI_to-MNI_xform.nii.gz')
    nib.save(mapping, mapping_file)
    reg_prealign_file = op.join(
        myafq.export("results_dir"),
        'sub-01_ses-01_dwi_desc-prealign_from-DWI_to-MNI_xform.npy')
    np.save(reg_prealign_file, np.eye(4))

    # Test ROI exporting:
    myafq.export("rois")
    assert op.exists(op.join(
        myafq.export("results_dir"),
        'ROIs',
        'sub-01_ses-01_dwi_space-subject_desc-CSTRinclude1_mask.json'))

    seg_sft = aus.SegmentedSFT.fromfile(
        myafq.export("bundles"))
    npt.assert_(len(seg_sft.get_bundle('SLF_R').streamlines) > 0)

    # Test bundles exporting:
    myafq.export("indiv_bundles")
    assert op.exists(op.join(
        myafq.export("results_dir"),
        'bundles',
        'sub-01_ses-01_dwi_space-RASMM_model-probCSD_algo-AFQ_desc-SLFL_tractography.trk'))  # noqa

    tract_profile_fname = myafq.export("profiles")
    tract_profiles = pd.read_csv(tract_profile_fname)

    assert tract_profiles.select_dtypes(include=[np.number]).sum().sum() != 0
    assert tract_profiles.shape == (400, 9)

    myafq.export("indiv_bundles_figures")
    assert op.exists(op.join(
        myafq.export("results_dir"),
        "viz_bundles",
        'sub-01_ses-01_dwi_space-RASMM_model-probCSD_algo-AFQ_desc-SLFLviz_dwi.html'))  # noqa

    assert op.exists(op.join(
        myafq.export("results_dir"),
        "viz_bundles",
        'sub-01_ses-01_dwi_space-RASMM_model-probCSD_algo-AFQ_desc-SLFLviz_dwi.html'))  # noqa

    # Before we run the CLI, we'll remove the bundles and ROI folders, to see
    # that the CLI generates them
    shutil.rmtree(op.join(myafq.export("results_dir"),
                          'bundles'))

    shutil.rmtree(op.join(myafq.export("results_dir"),
                          'ROIs'))
    os.remove(tract_profile_fname)

    # save memory
    results_dir = myafq.export("results_dir")
    del myafq
    gc.collect()

    # Test the CLI:
    print("Running the CLI:")

    # Set up config to use the same parameters as above:
    # ROI mask needs to be put in quotes in config
    tracking_params = dict(odf_model="CSD",
                           seed_mask="RoiImage()",
                           n_seeds=100,
                           random_seeds=True,
                           rng_seed=42)
    bundle_dict_as_str = (
        'BundleDict(["SLF_L", "SLF_R", "ARC_L", '
        '"ARC_R", "CST_L", "CST_R", "FP"])'
        '+ BundleDict({"LV1": {"start": '
        f'"{lv1_fname}", '
        '"space": "subject"}})')
    config = dict(
        BIDS_PARAMS=dict(
            bids_path=bids_path,
            preproc_pipeline='vistasoft'),
        DATA=dict(
            bundle_info=bundle_dict_as_str),
        SEGMENTATION=dict(
            scalars=[
                "dti_fa",
                "dti_md",
                "dti_ga",
                "dti_lt2",
                f"ImageFile('{t1_path_other}')",
                f"TemplateImage('{t1_path}')"]),
        TRACTOGRAPHY_PARAMS=tracking_params,
        SEGMENTATION_PARAMS=segmentation_params,
        CLEANING_PARAMS=clean_params)

    config_file = op.join(tmpdir, "afq_config.toml")
    with open(config_file, 'w') as ff:
        toml.dump(config, ff)

    cmd = f"pyAFQ -v {config_file}"
    completed_process = subprocess.run(
        cmd, shell=True, capture_output=True)
    if completed_process.returncode != 0:
        print(completed_process.stdout)
    print(completed_process.stderr)
    assert completed_process.returncode == 0
    # The tract profiles should already exist from the CLI Run:
    from_file = pd.read_csv(tract_profile_fname)

    assert from_file.shape == (400, 9)
    assert_series_equal(tract_profiles['dti_fa'], from_file['dti_fa'])

    # Make sure the CLI did indeed generate these:
    assert op.exists(op.join(
        results_dir,
        'ROIs',
        'sub-01_ses-01_dwi_space-subject_desc-SLFLinclude1_mask.json'))

    assert op.exists(op.join(
        results_dir,
        'bundles',
        'sub-01_ses-01_dwi_space-RASMM_model-probCSD_algo-AFQ_desc-SLFL_tractography.trk'))  # noqa
