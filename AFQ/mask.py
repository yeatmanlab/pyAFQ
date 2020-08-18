import numpy as np

import nibabel as nib

import AFQ.registration as reg

class _MaskFile(object):
    def __init__(self, suffix, scope=None):
        self.suffix = suffix
        self.scope = scope
        self.fnames = {}
    
    def get_path(self, bids_layout, subject, session):
        if session not in self.fnames:
            self.fnames[session] = {}
        self.fnames[session][subject] = bids_layout.get(
            subject=subject, session=session,
            extension='.nii.gz',
            suffix=self.suffix,
            return_type='filename',
            scope=self.scope)[0]
    
    def combine_illdefined(self, combine):
        raise TypeError((
            f"combine should be either 'or' or 'and',"
            f" you set combine to {combine}"))


class LabelledMaskFile(_MaskFile):
    def __init__(self, suffix, scope=None, inclusive_labels=None,
                 exclusive_labels=None, combine="or"):
        super.__init__(suffix, scope)
        self.ilabels = inclusive_labels
        self.elabels = exclusive_labels
        self.combine = combine

    def get_mask(self, dwi_data, dwi_affine, subject, session):
        mask_file = self.fnames[session][subject]
        mask_img = nib.load(mask_file)
        mask_data_orig = mask_img.get_fdata()
    
        # For different sets of labels, extract all the voxels that
        # have any / all of these values:
        if self.combine == "or":
            mask = np.zeros(mask_data_orig.shape, dtype=bool)
        elif self.combine == "and":
            mask = np.ones(mask_data_orig.shape, dtype=bool)
        else:
            super.combine_illdefined()
        if self.ilabels is not None:
            for label in self.ilabels:
                if self.combine == "or":
                    mask = np.logical_or(mask, (mask_data_orig == label))
                else:
                    mask = np.logical_and(mask, (mask_data_orig == label))
        if self.elabels is not None:
            for label in self.elabels:
                if self.combine == "or":
                    mask = np.logical_or(mask, (mask_data_orig != label))
                else:
                    mask = np.logical_and(mask, (mask_data_orig != label))

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
        super().__init__(suffix, scope)
        self.lb = lower_bound
        self.ub = upper_bound
