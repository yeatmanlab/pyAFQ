import numpy as np
import AFQ.recognition.other_bundles as abo

# Example data for testing
this_bundle_sls_sample = np.array([
    [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
    [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
])
other_bundle_sls_sample = np.array([
    [[0, 1, 2], [1, 2, 3], [2, 2, 2]],
    [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
])
img_sample = np.zeros((5, 5, 5))
node_thresh_sample = 1


def test_clean_by_other_density_map():
    cleaned_idx = abo.clean_by_other_density_map(
        this_bundle_sls_sample,
        other_bundle_sls_sample,
        node_thresh_sample,
        img_sample
    )
    assert isinstance(cleaned_idx, np.ndarray)
    assert cleaned_idx.shape[0] == this_bundle_sls_sample.shape[0]
    assert np.all(cleaned_idx == [False, True])


def test_clean_relative_to_other_core():
    for core in ['anterior', 'posterior', 'superior', 'inferior', 'right', 'left']:
        cleaned_idx_core = abo.clean_relative_to_other_core(
            core,
            this_bundle_sls_sample,
            other_bundle_sls_sample
        )

        assert isinstance(cleaned_idx_core, np.ndarray)
        assert cleaned_idx_core.shape[0] == this_bundle_sls_sample.shape[0]
        if core == "inferior":
            assert np.all(cleaned_idx_core == [False, True])
        else:
            assert np.all(cleaned_idx_core == [False, False])
