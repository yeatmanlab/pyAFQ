import numpy as np

import nibabel as nib

import AFQ.registration as reg


class _Mask(object):
    def __init__(self, shape, combine):
        self.combine = combine
        if self.combine == "or":
            self.mask = np.zeros(shape, dtype=bool)
        elif self.combine == "and":
            self.mask = np.ones(shape, dtype=bool)
        else:
            self.combine_illdefined()

    def combine_mask(self, other_mask):
        if self.combine == "or":
            self.mask = np.logical_or(self.mask, other_mask)
        elif self.combine == "and":
            self.mask = np.logical_and(self.mask, other_mask)
        else:
            self.combine_illdefined()

    def combine_illdefined(self):
        raise TypeError((
            f"combine should be either 'or' or 'and',"
            f" you set combine to {self.combine}"))

    def resample(self, data, affine, my_affine):
        if ((data is not None)
            and (affine is not None)
            and (data[..., 0].shape != self.mask.shape)):
            self.mask = np.round(reg.resample(self.mask.astype(float),
                                     data[..., 0],
                                     my_affine,
                                     affine)).astype(int)


class _MaskFile(object):
    def __init__(self, suffix, scope, combine):
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

    def get_path_data_affine(self, row):
        mask_file = self.fnames[row['ses']][row['subject']]
        mask_img = nib.load(mask_file)
        return mask_file, mask_img.get_fdata(), mask_img.affine


class LabelledMaskFile(_MaskFile):
    def __init__(self, suffix, scope=None, inclusive_labels=None,
                 exclusive_labels=None, combine="or"):
        super().__init__(suffix, scope)
        self.combine = combine
        self.ilabels = inclusive_labels
        self.elabels = exclusive_labels

    def get_mask(self, afq, row):
        dwi_data, _, dwi_img = afq._get_data_gtab(row)
        mask_file, mask_data_orig, mask_affine = \
            self.get_path_data_affine(row)

        # For different sets of labels, extract all the voxels that
        # have any / all of these values:
        mask = _Mask(mask_data_orig.shape, self.combine)
        if self.ilabels is not None:
            for label in self.ilabels:
                mask.combine_mask(mask_data_orig == label)
        if self.elabels is not None:
            for label in self.elabels:
                mask.combine_mask(mask_data_orig != label)

        # Resample to DWI data:
        mask.resample(dwi_data, dwi_img.affine, mask_affine)

        meta = dict(source=mask_file,
                    inclusive_labels=self.ilabels,
                    exclusive_lavels=self.elabels,
                    combined_with=self.combine)

        return mask.mask, meta


class ThresholdedMaskFile(_MaskFile):
    def __init__(self, suffix, scope=None, lower_bound=None,
                 upper_bound=None, combine="and"):
        super().__init__(suffix, scope)
        self.combine = combine
        self.lb = lower_bound
        self.ub = upper_bound

    def get_mask(self, afq, row):
        dwi_data, _, dwi_img = afq._get_data_gtab(row)
        mask_file, mask_data_orig, mask_affine = \
            self.get_path_data_affine(row)

        mask = _Mask(mask_data_orig.shape, self.combine)
        if self.ub is not None:
            mask.combine_mask(mask_data_orig < self.ub)
        if self.lb is not None:
            mask.combine_mask(mask_data_orig > self.lb)

        # Resample to DWI data:
        mask.resample(dwi_data, dwi_img.affine, mask_affine)

        meta = dict(source=mask_file,
                    upper_bound=self.ub,
                    lower_bound=self.lb,
                    combined_with=self.combine)

        return mask.mask, meta


class ThresholdedScalarMask(object):
    def __init__(self, scalar, lower_bound=None, upper_bound=None,
                 combine="and"):
        self.scalar_name = scalar
        self.combine = combine
        self.lb = lower_bound
        self.ub = upper_bound

    def find_path(self, bids_layout, subject, session):
        pass

    def get_mask(self, afq, row):
        dwi_data, _, dwi_img = afq._get_data_gtab(row)
        valid_scalars = list(afq._scalar_dict.keys())
        if self.scalar_name not in valid_scalars:
            raise RuntimeError((
                f"scalar should be one of"
                f" {', '.join(valid_scalars)}"
                f", you input {self.scalar_name}"))

        scalar_fname = afq._scalar_dict[self.scalar_name](afq, row)
        scalar_img = nib.load(scalar_fname)
        scalar_data = scalar_img.get_fdata()

        mask = _Mask(scalar_data.shape, self.combine)
        if self.ub is not None:
            mask.combine_mask(scalar_data < self.ub)
        if self.lb is not None:
            mask.combine_mask(scalar_data > self.lb)

        # Resample to DWI data:
        mask.resample(dwi_data, dwi_img.affine, scalar_img.affine)

        meta = dict(source=scalar_fname,
                    upper_bound=self.ub,
                    lower_bound=self.lb,
                    combined_with=self.combine)
        
        return mask.mask, meta
