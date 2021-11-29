import AFQ.api.bundle_dict as abd
from AFQ.tests.test_api import create_dummy_bids_path
from AFQ.api.group import GroupAFQ
import AFQ.data as afd
import pytest


def test_AFQ_custom_bundle_dict():
    bids_path = create_dummy_bids_path(3, 1)
    bundle_dict = abd.BundleDict()
    GroupAFQ(
        bids_path,
        preproc_pipeline="synthetic",
        bundle_info=bundle_dict)


def test_BundleDict():
    """
    Tests bundle dict
    """

    # test defaults
    afq_bundles = abd.BundleDict()

    # bundles restricted within hemisphere
    # NOTE: FA and FP cross midline so are removed
    # NOTE: all others generate two bundles
    num_hemi_bundles = (len(abd.BUNDLES)-2)*2

    # bundles that cross the midline
    num_whole_bundles = 2

    assert len(afq_bundles) == num_hemi_bundles + num_whole_bundles

    # Arcuate Fasciculus
    afq_bundles = abd.BundleDict(["ARC"])

    assert len(afq_bundles) == 2

    # Forceps Minor
    afq_bundles = abd.BundleDict(["FA"])

    assert len(afq_bundles) == 1

    # Cingulum Hippocampus
    # not included but exists in templates
    afq_bundles = abd.BundleDict(["HCC"])

    assert len(afq_bundles) == 2

    # Test "custom" bundle
    afq_templates = afd.read_templates()
    afq_bundles = abd.BundleDict({
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
        afq_bundles = abd.BundleDict(["VOF"])
        afq_bundles["VOF_R"]

    afq_bundles = abd.BundleDict(["VOF"], seg_algo="reco80")
    assert len(afq_bundles) == 2

    afq_bundles = abd.BundleDict(["whole_brain"], seg_algo="reco80")
    assert len(afq_bundles) == 1
