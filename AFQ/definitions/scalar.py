from AFQ.definitions.utils import Definition
from AFQ.definitions.mask import MaskFile

import AFQ.data as afd

import nibabel as nib
import os.path as op

from dipy.align import resample

# For scalar defintions, get_for_row should return:
# data, affine, meta
# additionally, each class should have a name parameter

__all__ = ["ScalarFile", "TemplateScalar"]


class ScalarMixin():
    def get_for_row(self, afq_object, row):
        scalar_file = afq_object._get_fname(
            row, f'_model-{self.name}.nii.gz')
        if not op.exists(scalar_file):
            scalar_data, meta = self.get_data(afq_object, row)

            afq_object.log_and_save_nii(
                nib.Nifti1Image(scalar_data, row['dwi_affine']),
                scalar_file)
            meta_fname = afq_object._get_fname(
                row, f'_model-{self.name}.json')
            afd.write_json(meta_fname, meta)
        return scalar_file


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
    my_scalar = ScalarFile(
        "my_scalar",
        "scalarSuffix",
        {"scope": "dmriprep"})
    api.AFQ(scalars=["dti_fa", "dti_md", my_scalar])
    """

    def __init__(self, name, suffix, filters={}):
        MaskFile.__init__(self, suffix, filters)
        self.name = name

    def get_data(self, afq_object, row):
        return self.fnames[row['ses']][row['subject']]


class TemplateScalar(ScalarMixin, Definition):
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
        "my_scalar", "path/to/my_scalar_in_MNI.nii.gz")
    api.AFQ(scalars=["dti_fa", "dti_md", my_scalar])
    """

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.img = nib.load(self.path)
        self.is_resampled = False

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_data(self, afq_object, row):
        if not self.is_resampled:
            self.img = resample(
                self.img.get_fdata(),
                afq_object.reg_template_img,
                self.img.affine,
                afq_object.reg_template_img.affine).get_fdata()
            self.is_resampled = True

        mapping = afq_object._mapping(row)

        scalar_data = afq_object._mapping(row).transform_inverse(self.img)

        return scalar_data, dict(source=self.path)
