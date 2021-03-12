from AFQ.definitions.utils import Definition
from AFQ.definitions.mask import MaskFile
import AFQ.registration as reg

# For scalar defintions, get_for_row should return:
# data, affine, meta
# additionally, each class should have a name parameter

__all__ = ["ScalarFile", "TemplateScalar"]


class ScalarFile(MaskFile):
    """
    Define a scalar based on a file for each subject.
    Assumes the scalar is already in subject space.

    Parameters
    ----------
    name : str
        name of the scalar.
    suffix : str
        suffix to pass to bids_layout.get() to identify the file.
    filters : str
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}

    Examples
    --------
    seed_mask = MaskFile(
        "WM_mask",
        {"scope"="dmriprep"})
    api.AFQ(tracking_params={"seed_mask": seed_mask,
                                "seed_threshold": 0.1})
    """

    def __init__(self, name, suffix, filters={}):
        MaskFile.__init__(self, suffix, filters)
        self.name = name


class TemplateScalar(Definition):
    """
    Define a scalar based on a template.
    This template will be transformed into subject space before use.

    Parameters
    ----------
    name : str
        name of the scalar.
    path : str
        path to the template.

    Examples
    --------
    my_scalar = TemplateScalar(
        "path/to/my_scalar_in_MNI.nii.gz")
    api.AFQ(scalars=["dti_fa", "dti_md", my_scalar])
    """

    def __init__(self, path, name):
        self.name = name
        self.path = path
        self.img = nib.load(self.path)
        self.is_resampled = False

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_for_row(self, afq_object, row):
        if not self.is_resampled:
            self.img = reg.resample(
                self.img.get_fdata(),
                afq_object.reg_template_img,
                self.img.affine,
                afq_object.reg_template_img.affine)
            self.is_resampled = True

        mapping = afq_object._mapping(row)

        mask_data = afq_object._mapping(row).transform_inverse(
            self.img.get_fdata())

        return mask_data, afq_object["dwi_affine"], dict(source=path)
