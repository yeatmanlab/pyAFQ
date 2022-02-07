import AFQ.data.s3bids as afs

import pytest
import s3fs
import shutil
import os.path as op

from glob import glob
from moto import mock_s3
from uuid import uuid4

import os

import botocore
import filecmp

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
    client = afs.get_s3_client()
    client.create_bucket(Bucket=TEST_BUCKET)
    fs.put(
        op.join(DATA_PATH, TEST_DATASET),
        "/".join([TEST_BUCKET, TEST_DATASET]),
        recursive=True,
    )


@mock_s3
def test_get_s3_client():
    client_anon = afs.get_s3_client(anon=True)
    assert isinstance(client_anon, botocore.client.BaseClient)
    assert client_anon.meta.service_model.service_id == "S3"
    assert client_anon.meta.config.signature_version == botocore.UNSIGNED

    client = afs.get_s3_client(anon=False)
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
        afs._get_matching_s3_keys(bucket=TEST_BUCKET, prefix=TEST_DATASET)
    )

    assert set(fnames) == set(matching_keys)


@mock_s3
def test_download_from_s3(temp_data_dir):
    s3_setup()

    test_dir = temp_data_dir

    matching_keys = list(
        afs._get_matching_s3_keys(bucket=TEST_BUCKET, prefix=TEST_DATASET)
    )

    first_json_file = [m for m in matching_keys if m.endswith(".json")][0]
    print(first_json_file)
    print(TEST_BUCKET)

    afs._download_from_s3(
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

    study = afs.S3BIDSStudy(
        study_id="test",
        bucket=TEST_BUCKET,
        s3_prefix=TEST_DATASET,
        anon=False,
        random_seed=42,
        subjects=5,
    )

    assert len(study.subjects) == 5

    study = afs.S3BIDSStudy(
        study_id="test",
        bucket=TEST_BUCKET,
        s3_prefix=TEST_DATASET,
        anon=False,
        random_seed=42,
    )

    assert len(study.subjects) == 1

    study = afs.S3BIDSStudy(
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
