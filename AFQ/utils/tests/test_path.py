from AFQ.utils.path import drop_extension


def test_drop_extension():
    assert "/my/.example/.path/to_file" == drop_extension(
        "/my/.example/.path/to_file.nii.gz")
