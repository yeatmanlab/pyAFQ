import botocore
import filecmp
import numpy as np
import numpy.testing as npt
import os
import os.path as op
import pytest
import s3fs
import shutil

from glob import glob
from moto import mock_s3
from uuid import uuid4

import AFQ.data as afd
import nibabel as nib

DATA_PATH = op.join(op.abspath(op.dirname(__file__)), "data/mocks3")
TEST_BUCKET = "test-bucket"
TEST_DATASET = "ds000102-mimic"


@pytest.fixture
def temp_data_dir():
    test_dir = str(uuid4())
    os.mkdir(test_dir)

    yield test_dir

    shutil.rmtree(test_dir)


@mock_s3
def s3_setup():
    """pytest fixture to put test_data directory on mock_s3"""
    fs = s3fs.S3FileSystem()
    client = afd.get_s3_client()
    client.create_bucket(Bucket=TEST_BUCKET)
    fs.put(
        op.join(DATA_PATH, TEST_DATASET),
        "/".join([TEST_BUCKET, TEST_DATASET]),
        recursive=True,
    )


@mock_s3
def test_get_s3_client():
    client_anon = afd.get_s3_client(anon=True)
    assert isinstance(client_anon, botocore.client.BaseClient)
    assert client_anon.meta.service_model.service_id == "S3"
    assert client_anon.meta.config.signature_version == botocore.UNSIGNED

    client = afd.get_s3_client(anon=False)
    assert isinstance(client, botocore.client.BaseClient)
    assert client.meta.service_model.service_id == "S3"
    assert isinstance(client.meta.config.signature_version, str)


@mock_s3
def test_get_matching_s3_keys():
    s3_setup()

    fnames = []
    for pattern in ["**", "*/.*", "*/.*/.*", "*/.*/**"]:
        fnames += [
            s for s in glob(op.join(DATA_PATH, pattern),
                            recursive=True) if op.isfile(s)
        ]

    fnames = [s.replace(DATA_PATH + "/", "") for s in fnames]

    matching_keys = list(
        afd._get_matching_s3_keys(bucket=TEST_BUCKET, prefix=TEST_DATASET)
    )

    assert set(fnames) == set(matching_keys)


@mock_s3
def test_download_from_s3(temp_data_dir):
    s3_setup()

    test_dir = temp_data_dir

    matching_keys = list(
        afd._get_matching_s3_keys(bucket=TEST_BUCKET, prefix=TEST_DATASET)
    )

    first_json_file = [m for m in matching_keys if m.endswith(".json")][0]
    print(first_json_file)
    print(TEST_BUCKET)

    afd._download_from_s3(
        fname=op.join(test_dir, "test_file"),
        bucket=TEST_BUCKET,
        key=first_json_file,
        anon=False,
    )

    assert op.isfile(op.join(test_dir, "test_file"))


@mock_s3
def test_S3BIDSStudy(temp_data_dir):
    s3_setup()

    test_dir = temp_data_dir

    study = afd.S3BIDSStudy(
        study_id="test",
        bucket=TEST_BUCKET,
        s3_prefix=TEST_DATASET,
        anon=False,
        random_seed=42,
        subjects=5,
    )

    assert len(study.subjects) == 5

    study = afd.S3BIDSStudy(
        study_id="test",
        bucket=TEST_BUCKET,
        s3_prefix=TEST_DATASET,
        anon=False,
        random_seed=42,
    )

    assert len(study.subjects) == 1

    study = afd.S3BIDSStudy(
        study_id="test",
        bucket=TEST_BUCKET,
        s3_prefix=TEST_DATASET,
        anon=False,
        subjects="all",
    )

    assert len(study.subjects) == len(study._all_subjects)

    study.download(test_dir)
    study._download_non_sub_keys(test_dir)

    s0 = study.subjects[0]
    download_files = list(s0.files["raw"].values())
    ref_dir = op.abspath(op.join(DATA_PATH, TEST_DATASET))
    match, mismatch, errors = filecmp.cmpfiles(
        ref_dir, test_dir, download_files, shallow=False
    )

    assert not mismatch
    assert not errors

    try:
        test_dir2 = str(uuid4())
        os.mkdir(test_dir2)

        study.download(test_dir2)
        match, mismatch, errors = filecmp.cmpfiles(
            ref_dir, test_dir2, download_files, shallow=False
        )
        assert not mismatch
        assert not errors
    finally:
        shutil.rmtree(test_dir2)


def test_aal_to_regions():
    atlas = np.zeros((20, 20, 20, 5))
    # Just one region:
    atlas[0, 0, 0, 0] = 1
    regions = ["leftfrontal"]
    idx = afd.aal_to_regions(regions, atlas=atlas)
    npt.assert_equal(idx, np.array(np.where(atlas[..., 0] == 1)).T)

    # More than one region:
    atlas[1, 1, 1, 0] = 2
    regions = ["leftfrontal", "rightfrontal"]
    idx = afd.aal_to_regions(regions, atlas=atlas)

    npt.assert_equal(
        idx, np.array(np.where((atlas[..., 0] == 1) | (atlas[..., 0] == 2))).T
    )

    # Use one of the additional volumes
    atlas[1, 1, 1, 1] = 1
    regions = ["leftfrontal", "rightfrontal", "cstsuperior"]
    idx = afd.aal_to_regions(regions, atlas=atlas)

    npt.assert_equal(
        idx,
        np.array(
            np.where((atlas[..., 0] == 1) | (atlas[..., 0] == 2) | (atlas[..., 1] == 1))
        ).T,
    )


def test_bundles_to_aal():
    atlas = np.zeros((20, 20, 20, 5))

    atlas[0, 0, 0, 0] = 1

    targets = afd.bundles_to_aal(["ATR_L"], atlas)
    npt.assert_equal(targets, [[np.array(np.where(atlas[..., 0] == 1)).T, None]])

    atlas[0, 0, 1, 0] = 2

    targets = afd.bundles_to_aal(["ATR_L", "ATR_R"], atlas)
    npt.assert_equal(
        targets,
        [
            [np.array(np.where(atlas[..., 0] == 1)).T, None],
            [np.array(np.where(atlas[..., 0] == 2)).T, None],
        ],
    )

    targets = afd.bundles_to_aal([], atlas)
    assert len(targets) == 0

    targets = afd.bundles_to_aal(["HCC_L"], atlas)
    assert len(targets) == 1
    npt.assert_equal(targets, [[None, None]])

    targets = afd.bundles_to_aal(["VOF"], atlas)
    assert len(targets) == 1
    npt.assert_equal(targets, [[None, None]])

    
def test_read_roi():
    aff1 = np.eye(4)
    template = nib.Nifti1Image(np.ones((10, 10, 10)), aff1)
    aff2 = aff1[:]
    aff2[0, 0] = -1
    roi = nib.Nifti1Image(np.zeros((10, 10, 10)), aff2)
    img = afd.read_resample_roi(roi, resample_to=template)
    npt.assert_equal(img.affine, template.affine)
