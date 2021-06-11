import numpy as np

import nibabel as nib
from dipy.segment.mask import median_otsu

from dipy.align import resample
import AFQ.utils.volume as auv
from AFQ.definitions.utils import Definition, find_file
from AFQ.tasks.decorators import as_img


__all__ = ["MaskFile", "FullMask", "RoiMask", "B0Mask", "LabelledMaskFile",
           "ThresholdedMaskFile", "ScalarMask", "ThresholdedScalarMask"]


def _resample_mask(mask_data, dwi_data, mask_affine, dwi_affine):
    '''
    Helper function
    Resamples mask to dwi if necessary
    '''
    mask_type = mask_data.dtype
    if ((dwi_data is not None)
        and (dwi_affine is not None)
            and (dwi_data[..., 0].shape != mask_data.shape)):
        return np.round(resample(
            mask_data.astype(float),
            dwi_data[..., 0],
            mask_affine,
            dwi_affine).get_fdata()).astype(mask_type)
    else:
        return mask_data


class CombineMaskMixin(object):
    """
    Helper Class
    Useful for making a mask by combining different conditions
    """

    def __init__(self, combine):
        self.combine = combine.lower()

    def reset_mask_draft(self, shape):
        if self.combine == "or":
            self.mask_draft = np.zeros(shape, dtype=bool)
        elif self.combine == "and":
            self.mask_draft = np.ones(shape, dtype=bool)
        else:
            self.combine_illdefined()

    def __mul__(self, other_mask):
        if self.combine == "or":
            return np.logical_or(self.mask_draft, other_mask)
        elif self.combine == "and":
            return np.logical_and(self.mask_draft, other_mask)
        else:
            self.combine_illdefined()

    def combine_illdefined(self):
        raise TypeError((
            f"combine should be either 'or' or 'and',"
            f" you set combine to {self.combine}"))


class MaskFile(Definition):
    """
    Define a mask based on a file.
    Does not apply any labels or thresholds;
    Generates mask with floating point data.
    Useful for seed and stop masks, where threshold can be applied
    after interpolation (see example).

    Parameters
    ----------
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
        {"scope":"dmriprep"})
    api.AFQ(tracking_params={"seed_mask": seed_mask,
                                "seed_threshold": 0.1})
    """

    def __init__(self, suffix, filters={}):
        self.suffix = suffix
        self.filters = filters
        self.fnames = {}

    def find_path(self, bids_layout, from_path, subject, session):
        if session not in self.fnames:
            self.fnames[session] = {}

        nearest_mask = find_file(
            bids_layout, from_path, self.filters, self.suffix, session,
            subject)

        self.fnames[session][subject] = nearest_mask

    def get_path_data_affine(self, subses_dict):
        mask_file = self.fnames[subses_dict['ses']][subses_dict['subject']]
        mask_img = nib.load(mask_file)
        return mask_file, mask_img.get_fdata(), mask_img.affine

    # This function is set up to be overriden by other masks
    def apply_conditions(self, mask_data_orig, mask_file):
        return mask_data_orig, dict(source=mask_file)

    def get_mask_getter(self, in_data=False):
        @as_img
        def mask_getter(subses_dict, dwi_affine):
            # Load data
            mask_file, mask_data_orig, mask_affine = \
                self.get_path_data_affine(subses_dict)

            # Apply any conditions on the data
            mask_data, meta = self.apply_conditions(mask_data_orig, mask_file)

            # Resample to DWI data:
            mask_data = _resample_mask(
                mask_data,
                nib.load(subses_dict["dwi_file"]).get_fdata(),
                mask_affine,
                dwi_affine)
            return mask_data, meta
        return mask_getter


class FullMask(Definition):
    """
    Define a mask which covers a full volume.

    Examples
    --------
    brain_mask = FullMask()
    """

    def __init__(self):
        pass

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_mask_getter(self, in_data=False):
        @as_img
        def mask_getter(subses_dict, dwi_affine):
            return np.ones(nib.load(
                subses_dict["dwi_file"]).get_fdata()[..., 0].shape),\
                dict(source="Entire Volume")
        return mask_getter


class RoiMask(Definition):
    """
    Define a mask which is all ROIs or'd together.

    Examples
    --------
    seed_mask = RoiMask()
    api.AFQ(tracking_params={"seed_mask": seed_mask})
    """

    def __init__(self, use_presegment=False):
        self.use_presegment = use_presegment
        pass

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_mask_getter(self, in_data=False):
        @as_img
        def mask_getter(subses_dict, dwi_affine, mapping_imap,
                        bundle_dict, segmentation_params):
            mask_data = None
            if self.use_presegment:
                bundle_dict = \
                    segmentation_params["presegment_bundle_dict"]
            else:
                bundle_dict = bundle_dict

            for bundle_name, bundle_info in bundle_dict.items():
                for idx, roi in enumerate(bundle_info['ROIs']):
                    if bundle_dict[bundle_name]['rules'][idx]:
                        warped_roi = auv.transform_inverse_roi(
                            roi,
                            mapping_imap["mapping"],
                            bundle_name=bundle_name)

                        if mask_data is None:
                            mask_data = np.zeros(warped_roi.shape)
                        mask_data = np.logical_or(
                            mask_data,
                            warped_roi.astype(bool))
            return mask_data, dict(source="ROIs")
        return mask_getter


class B0Mask(Definition):
    """
    Define a mask using b0 and dipy's median_otsu.

    Parameters
    ----------
    median_otsu_kwargs: dict, optional
        Optional arguments to pass into dipy's median_otsu.
        Default: {}

    Examples
    --------
    brain_mask = B0Mask()
    api.AFQ(brain_mask=brain_mask)
    """

    def __init__(self, median_otsu_kwargs={}):
        self.median_otsu_kwargs = median_otsu_kwargs

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_mask_getter(self, in_data=False):
        if in_data:
            @as_img
            def mask_getter(subses_dict, dwi_affine, b0_file):
                mean_b0_img = nib.load(b0_file)
                mean_b0 = mean_b0_img.get_fdata()
                _, mask_data = median_otsu(mean_b0, **self.median_otsu_kwargs)
                return mask_data, dict(
                    source=b0_file,
                    technique="median_otsu applied to b0",
                    median_otsu_kwargs=self.median_otsu_kwargs)
        else:
            @as_img
            def mask_getter(subses_dict, dwi_affine, data_imap):
                mean_b0_img = nib.load(data_imap["b0_file"])
                mean_b0 = mean_b0_img.get_fdata()
                _, mask_data = median_otsu(mean_b0, **self.median_otsu_kwargs)
                return mask_data, dict(
                    source=data_imap["b0_file"],
                    technique="median_otsu applied to b0",
                    median_otsu_kwargs=self.median_otsu_kwargs)
        return mask_getter


class LabelledMaskFile(MaskFile, CombineMaskMixin):
    """
    Define a mask based on labels in a file.

    Parameters
    ----------
    suffix : str
        suffix to pass to bids_layout.get() to identify the file.
    filters : str
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    inclusive_labels : list of ints, optional
        The labels from the file to include from the boolean mask.
        If None, no inclusive labels are applied.
    exclusive_labels : lits of ints, optional
        The labels from the file to exclude from the boolean mask.
        If None, no exclusive labels are applied.
        Default: None.
    combine : str, optional
        How to combine the boolean masks generated by inclusive_labels
        and exclusive_labels. If "and", they will be and'd together.
        If "or", they will be or'd.
        Note: in this class, you will most likely want to either set
        inclusive_labels or exclusive_labels, not both,
        so combine will not matter.
        Default: "or"

    Examples
    --------
    brain_mask = LabelledMaskFile(
        "aseg",
        {"scope": "dmriprep"},
        exclusive_labels=[0])
    api.AFQ(brain_mask=brain_mask)
    """

    def __init__(self, suffix, filters={}, inclusive_labels=None,
                 exclusive_labels=None, combine="or"):
        MaskFile.__init__(self, suffix, filters)
        CombineMaskMixin.__init__(self, combine)
        self.inclusive_labels = inclusive_labels
        self.exclusive_labels = exclusive_labels

    # overrides MaskFile
    def apply_conditions(self, mask_data_orig, mask_file):
        # For different sets of labels, extract all the voxels that
        # have any / all of these values:
        self.reset_mask_draft(mask_data_orig.shape)
        if self.inclusive_labels is not None:
            for label in self.inclusive_labels:
                self.mask_draft = self * (mask_data_orig == label)
        if self.exclusive_labels is not None:
            for label in self.exclusive_labels:
                self.mask_draft = self * (mask_data_orig != label)

        meta = dict(source=mask_file,
                    inclusive_labels=self.inclusive_labels,
                    exclusive_lavels=self.exclusive_labels,
                    combined_with=self.combine)
        return self.mask_draft, meta


class ThresholdedMaskFile(MaskFile, CombineMaskMixin):
    """
    Define a mask based on thresholding a file.
    Note that this should not be used to directly make a seed mask
    or a stop mask. In those cases, consider thresholding after
    interpolation, as in the example for MaskFile.

    Parameters
    ----------
    suffix : str
        suffix to pass to bids_layout.get() to identify the file.
    filters : str
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    lower_bound : float, optional
        Lower bound to generate boolean mask from data in the file.
        If None, no lower bound is applied.
        Default: None.
    upper_bound : float, optional
        Upper bound to generate boolean mask from data in the file.
        If None, no upper bound is applied.
        Default: None.
    combine : str, optional
        How to combine the boolean masks generated by lower_bound
        and upper_bound. If "and", they will be and'd together.
        If "or", they will be or'd.
        Default: "and"

    Examples
    --------
    brain_mask = ThresholdedMaskFile(
        "brain_mask",
        {"scope":"dmriprep"},
        lower_bound=0.1)
    api.AFQ(brain_mask=brain_mask)
    """

    def __init__(self, suffix, filters={}, lower_bound=None,
                 upper_bound=None, combine="and"):
        MaskFile.__init__(self, suffix, filters)
        CombineMaskMixin.__init__(self, combine)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    # overrides MaskFile
    def apply_conditions(self, mask_data_orig, mask_file):
        # Apply thresholds
        self.reset_mask_draft(mask_data_orig.shape)
        if self.upper_bound is not None:
            self.mask_draft = self * (mask_data_orig < self.upper_bound)
        if self.lower_bound is not None:
            self.mask_draft = self * (mask_data_orig > self.lower_bound)

        meta = dict(source=mask_file,
                    upper_bound=self.upper_bound,
                    lower_bound=self.lower_bound,
                    combined_with=self.combine)
        return self.mask_draft, meta


class ScalarMask(Definition):
    """
    Define a mask based on a scalar.
    Does not apply any labels or thresholds;
    Generates mask with floating point data.
    Useful for seed and stop masks, where threshold can be applied
    after interpolation (see example).

    Parameters
    ----------
    scalar : str
        Scalar to threshold.
        Can be one of "dti_fa", "dti_md", "dki_fa", "dki_md".

    Examples
    --------
    seed_mask = ScalarMask(
        "dti_fa",
        scope="dmriprep")
    api.AFQ(tracking_params={
        "seed_mask": seed_mask,
        "seed_threshold": 0.2})
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    # overrides MaskFile
    def get_mask_getter(self, in_data=False):
        if in_data:
            raise ValueError(
                "Brain mask cannot be a built-in scalar, "
                + "because the brain mask is used to calculate scalars")

        def mask_getter(subses_dict, dwi_affine, data_imap):
            scalar_file = data_imap[self.scalar + "_file"]
            return nib.load(scalar_file), dict(FromScalar=self.scalar)
        return mask_getter


class ThresholdedScalarMask(ThresholdedMaskFile, ScalarMask):
    """
    Define a mask based on thresholding a scalar mask.
    Note that this should not be used to directly make a seed mask
    or a stop mask. In those cases, consider thresholding after
    interpolation, as in the example for ScalarMask.

    Parameters
    ----------
    scalar : str
        Scalar to threshold.
        Can be one of "dti_fa", "dti_md", "dki_fa", "dki_md".
    lower_bound : float, optional
        Lower bound to generate boolean mask from data in the file.
        If None, no lower bound is applied.
        Default: None.
    upper_bound : float, optional
        Upper bound to generate boolean mask from data in the file.
        If None, no upper bound is applied.
        Default: None.
    combine : str, optional
        How to combine the boolean masks generated by lower_bound
        and upper_bound. If "and", they will be and'd together.
        If "or", they will be or'd.
        Default: "and"

    Examples
    --------
    seed_mask = ThresholdedScalarMask(
        "dti_fa",
        lower_bound=0.2)
    api.AFQ(tracking_params={"seed_mask": seed_mask})
    """

    def __init__(self, scalar, lower_bound=None, upper_bound=None,
                 combine="and"):
        self.scalar = scalar
        CombineMaskMixin.__init__(self, combine)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class PFTMask(Definition):
    """
    Define a mask for use in PFT tractography. Only use
    if tracker set to 'pft' in tractography.

    Parameters
    ----------
    WM_probseg : MaskFile
        White matter segmentation file.
    GM_probseg : MaskFile
        Gray matter segmentation file.
    CSF_probseg : MaskFile
        Corticospinal fluid segmentation file.

    Examples
    --------
    stop_mask = PFTMask(
        afm.MaskFile("WMprobseg"),
        afm.MaskFile("GMprobseg"),
        afm.MaskFile("CSFprobseg"))
    api.AFQ(tracking_params={
        "stop_mask": stop_mask,
        "stop_threshold": "CMC",
        "tracker": "pft"})
    """

    def __init__(self, WM_probseg, GM_probseg, CSF_probseg):
        self.probsegs = (WM_probseg, GM_probseg, CSF_probseg)

    def find_path(self, bids_layout, from_path, subject, session):
        for probseg in self.probsegs:
            probseg.find_path(bids_layout, from_path, subject, session)

    def get_mask_getter(self, in_data=False):
        probseg_funcs = []
        for probseg in self.probsegs:
            probseg_funcs.append(probseg.get_mask_getter())
        return probseg_funcs


# class CombinedMask(Definition, CombineMaskMixin):
#     """  # TODO: can this be done in current system?
#     Define a mask by combining other masks.

#     Parameters
#     ----------
#     mask_list : list of Masks with find_path and get_for_subses functions
#         List of masks to combine. All find_path methods will be called
#         when this find_path method is called. All get_for_subses methods will
#         be called and combined when this get_for_subses method is called.
#     combine : str, optional
#         How to combine the boolean masks generated by mask_list.
#         If "and", they will be and'd together.
#         If "or", they will be or'd.
#         Default: "and"

#     Examples
#     --------
#     seed_mask = CombinedMask(
#         [ThresholdedScalarMask(
#             "dti_fa",
#             lower_bound=0.2),
#         ThresholdedScalarMask(
#             "dti_md",
#             upper_bound=0.002)])
#     api.AFQ(tracking_params={"seed_mask": seed_mask})
#     """

#     def __init__(self, mask_list, combine="and"):
#         CombineMaskMixin.__init__(self, combine)
#         self.mask_list = mask_list

#     def find_path(self, bids_layout, from_path, subject, session):
#         for mask in self.mask_list:
#             mask.find_path(bids_layout, from_path, subject, session)

#     def get_for_subses(self):
#         self.mask_draft = None
#         metas = []
#         for mask in self.mask_list:
#             next_mask, next_affine, next_meta = mask.get_for_subses(
#                 afq_object, row)
#             if self.mask_draft is None:
#                 self.reset_mask_draft(next_mask.shape)
#             else:
#                 self.mask_draft = self * (next_mask)
#             metas.append(next_meta)

#         meta = dict(sources=metas,
#                     combined_with=self.combine)

#         return self.mask_draft, next_affine, meta
