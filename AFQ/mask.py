import numpy as np

import nibabel as nib

import AFQ.registration as reg


class _Mask(object):
    def __init__(self, combine):
        self.combine = combine

    def get_empty_mask(self, shape):
        if self.combine == "or":
            return np.zeros(shape, dtype=bool)
        elif self.combine == "and":
            return np.ones(shape, dtype=bool)
        else:
            self.combine_illdefined()

    def combine_mask(self, mask, other_mask):
        if self.combine == "or":
            return np.logical_or(mask, other_mask)
        elif self.combine == "and":
            return np.logical_and(mask, other_mask)
        else:
            self.combine_illdefined()

    def combine_illdefined(self):
        raise TypeError((
            f"combine should be either 'or' or 'and',"
            f" you set combine to {self.combine}"))


class _MaskFile(_Mask):
    def __init__(self, suffix, scope, combine):
        super().__init__(combine)
        self.suffix = suffix
        self.scope = scope
        self.fnames = {}

    def find_path(self, bids_layout, subject, session):
        if session not in self.fnames:
            self.fnames[session] = {}
        self.fnames[session][subject] = bids_layout.get(
            subject=subject, session=session,
            extension='.nii.gz',
            suffix=self.suffix,
            return_type='filename',
            scope=self.scope)[0]

    def get_path(self, subject, session):
        return self.fnames[session][subject]


class LabelledMaskFile(_MaskFile):
    def __init__(self, suffix, scope=None, inclusive_labels=None,
                 exclusive_labels=None, combine="or"):
        super().__init__(suffix, scope, combine)
        self.ilabels = inclusive_labels
        self.elabels = exclusive_labels

    def get_mask(self, dwi_data, dwi_affine, subject, session):
        mask_file = super().get_path(subject, session)
        mask_img = nib.load(mask_file)
        mask_data_orig = mask_img.get_fdata()

        # For different sets of labels, extract all the voxels that
        # have any / all of these values:
        mask = super().get_empty_mask(mask_data_orig.shape, self.combine)
        if self.ilabels is not None:
            for label in self.ilabels:
                mask = super().combine_mask(mask, mask_data_orig == label)
        if self.elabels is not None:
            for label in self.elabels:
                mask = super().combine_mask(mask, mask_data_orig != label)

        # Resample to DWI data:
        mask = np.round(reg.resample(mask.astype(float),
                                     dwi_data[..., 0],
                                     mask_img.affine,
                                     dwi_affine)).astype(int)
        meta = dict(source=mask_file,
                    inclusive_labels=self.ilabels,
                    exclusive_lavels=self.elabels,
                    combined_with=self.combine)

        return mask, meta


class ThresholdedMaskFile(_MaskFile):
    def __init__(self, suffix, scope=None, lower_bound=None,
                 upper_bound=None, combine="and"):
        super().__init__(suffix, scope, combine)
        self.lb = lower_bound
        self.ub = upper_bound

    def get_mask(self, dwi_data, dwi_affine, subject, session):
        mask_file = super().get_path(subject, session)
        mask_data = nib.load(mask_file).get_fdata()
        mask = super().get_empty_mask(mask_data.shape)

        if self.ub is not None:
            mask = super().combine_mask(mask, mask_data < self.ub)
        if self.lb is not None:
            mask = super().combine_mask(mask, mask_data > self.lb)

        meta = dict(source=mask_file,
                    upper_bound=self.ub,
                    lower_bound=self.lb,
                    combined_with=self.combine)
