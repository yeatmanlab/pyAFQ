import numpy as np
import logging

import nibabel as nib
from dipy.segment.mask import median_otsu

from dipy.align import resample
import AFQ.utils.volume as auv
from AFQ.definitions.utils import Definition, find_file, name_from_path


__all__ = [
    "ImageFile", "FullImage", "RoiImage", "B0Image", "LabelledImageFile",
    "ThresholdedImageFile", "ScalarImage", "ThresholdedScalarImage",
    "TemplateImage"]


logger = logging.getLogger('AFQ')


def _resample_image(image_data, dwi_data, image_affine, dwi_affine):
    '''
    Helper function
    Resamples image to dwi if necessary
    '''
    image_type = image_data.dtype
    if ((dwi_data is not None)
        and (dwi_affine is not None)
            and (dwi_data[..., 0].shape != image_data.shape)):
        return np.round(resample(
            image_data.astype(float),
            dwi_data[..., 0],
            image_affine,
            dwi_affine).get_fdata()).astype(image_type)
    else:
        return image_data


class ImageDefinition(Definition):
    '''
    All Image Definitions should inherit this.
    '''

    def get_name(self):
        raise NotImplementedError("Please implement a get_name method")

    def get_image_getter(self, task_name):
        raise NotImplementedError(
            "Please implement a get_image_getter method")

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        raise NotImplementedError(
            "Please implement a get_image_direct method")


class CombineImageMixin(object):
    """
    Helper Class
    Useful for making an image by combining different conditions
    """

    def __init__(self, combine):
        self.combine = combine.lower()

    def reset_image_draft(self, shape):
        if self.combine == "or":
            self.image_draft = np.zeros(shape, dtype=bool)
        elif self.combine == "and":
            self.image_draft = np.ones(shape, dtype=bool)
        else:
            self.combine_illdefined()

    def __mul__(self, other_image):
        if self.combine == "or":
            return np.logical_or(self.image_draft, other_image)
        elif self.combine == "and":
            return np.logical_and(self.image_draft, other_image)
        else:
            self.combine_illdefined()

    def combine_illdefined(self):
        raise TypeError((
            f"combine should be either 'or' or 'and',"
            f" you set combine to {self.combine}"))


class ImageFile(ImageDefinition):
    """
    Define an image based on a file.
    Does not apply any labels or thresholds;
    Generates image with floating point data.
    Useful for seed and stop images, where threshold can be applied
    after interpolation (see example).

    Parameters
    ----------
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}

    Examples
    --------
    seed_image = ImageFile(
        suffix="WM",
        filters={"scope":"dmriprep"})
    api.GroupAFQ(tracking_params={"seed_image": seed_image,
                                "seed_threshold": 0.1})
    """

    def __init__(self, path=None, suffix=None, filters={}):
        if path is None and suffix is None:
            raise ValueError((
                "One of `path` or `suffix` must set to "
                "a value other than None."))

        if path is not None:
            self._from_path = True
            self.fname = path
        else:
            self._from_path = False
            self.suffix = suffix
            self.filters = filters
            self.fnames = {}

    def find_path(self, bids_layout, from_path, subject, session):
        if self._from_path:
            return
        if session not in self.fnames:
            self.fnames[session] = {}

        nearest_image = find_file(
            bids_layout, from_path, self.filters, self.suffix, session,
            subject)

        self.fnames[session][subject] = nearest_image

    def get_path_data_affine(self, bids_info):
        if self._from_path:
            image_file = self.fname
        else:
            image_file = self.fnames[
                bids_info['session']][bids_info['subject']]
        image_img = nib.load(image_file)
        return image_file, image_img.get_fdata(), image_img.affine

    # This function is set up to be overriden by other images
    def apply_conditions(self, image_data_orig, image_file):
        return image_data_orig, dict(source=image_file)

    def get_name(self):
        return name_from_path(self.fname) if self._from_path else self.suffix

    def get_image_getter(self, task_name):
        def image_getter(dwi, bids_info):
            # Load data
            image_file, image_data_orig, image_affine = \
                self.get_path_data_affine(bids_info)

            # Apply any conditions on the data
            image_data, meta = self.apply_conditions(
                image_data_orig, image_file)

            # Resample to DWI data:
            dwi_img = nib.load(dwi)
            image_data = _resample_image(
                image_data,
                dwi_img.get_fdata(),
                image_affine,
                dwi_img.affine)
            return nib.Nifti1Image(
                image_data.astype(np.float32),
                dwi_img.affine), meta
        return image_getter

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        return self.get_image_getter("direct")(
            dwi, bids_info)


class FullImage(ImageDefinition):
    """
    Define an image which covers a full volume.

    Examples
    --------
    brain_image_definition = FullImage()
    """

    def __init__(self):
        pass

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_name(self):
        return "entire_volume"

    def get_image_getter(self, task_name):
        def image_getter(dwi):
            dwi_img = nib.load(dwi)
            return nib.Nifti1Image(
                np.ones(dwi_img.get_fdata()[..., 0].shape, dtype=np.float32),
                dwi_img.affine), dict(source="Entire Volume")
        return image_getter

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        return self.get_image_getter("direct")(dwi)


class RoiImage(ImageDefinition):
    """
    Define an image which is all include ROIs or'd together.

    Parameters
    ----------
    use_presegment : bool
        Whether to use presegment bundle dict from segmentation params
        to get ROIs.
    use_endpoints : bool
        Whether to use the endpoints ("start" and "end") instead of the
        include ROIs to generate the image.

    Examples
    --------
    seed_image = RoiImage()
    api.GroupAFQ(tracking_params={"seed_image": seed_image})
    """

    def __init__(self,
                 use_waypoints=True,
                 use_presegment=False,
                 use_endpoints=False):
        self.use_waypoints = use_waypoints
        self.use_presegment = use_presegment
        self.use_endpoints = use_endpoints
        if not np.logical_or(self.use_waypoints, np.logical_or(
                self.use_endpoints, self.use_presegment)):
            raise ValueError((
                "One of use_waypoints, use_presegment, "
                "use_endpoints, must be True"))

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_name(self):
        return "roi"

    def get_image_getter(self, task_name):
        def _image_getter_helper(dwi_affine, mapping,
                                 data_imap, segmentation_params):
            image_data = None
            bundle_dict = data_imap["bundle_dict"]
            if self.use_presegment:
                bundle_dict = \
                    segmentation_params["presegment_bundle_dict"]
            else:
                bundle_dict = bundle_dict

            for bundle_name, bundle_info in bundle_dict.items():
                rois = []
                if self.use_endpoints:
                    rois.extend(
                        [bundle_info[end_type] for end_type in
                            ["start", "end"] if end_type in bundle_info])
                if self.use_waypoints:
                    rois.extend(bundle_info['include']
                                if 'include' in bundle_info else [])
                for roi in rois:
                    if "space" not in bundle_info\
                        or bundle_info[
                            "space"] == "template":
                        warped_roi = auv.transform_inverse_roi(
                            roi,
                            mapping,
                            bundle_name=bundle_name)
                    else:
                        warped_roi = roi.get_fdata()

                    if image_data is None:
                        image_data = np.zeros(warped_roi.shape)
                    image_data = np.logical_or(
                        image_data,
                        warped_roi.astype(bool))
            return nib.Nifti1Image(
                image_data.astype(np.float32),
                dwi_affine), dict(source="ROIs")

        if task_name == "mapping":
            def image_getter(
                    dwi_affine, mapping,
                    data_imap, segmentation_params):
                return _image_getter_helper(
                    dwi_affine, mapping,
                    data_imap, segmentation_params)
        else:
            def image_getter(
                    dwi_affine, mapping_imap,
                    data_imap, segmentation_params):
                return _image_getter_helper(
                    dwi_affine, mapping_imap["mapping"],
                    data_imap, segmentation_params)
        return image_getter

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        raise ValueError((
            "RoiImage cannot be used in this context, as they"
            "require later derivatives to be calculated"))


class B0Image(ImageDefinition):
    """
    Define an image using b0 and dipy's median_otsu.

    Parameters
    ----------
    median_otsu_kwargs: dict, optional
        Optional arguments to pass into dipy's median_otsu.
        Default: {}

    Examples
    --------
    brain_image_definition = B0Image()
    api.GroupAFQ(brain_image_definition=brain_image_definition)
    """

    def __init__(self, median_otsu_kwargs={}):
        self.median_otsu_kwargs = median_otsu_kwargs

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_name(self):
        return "b0"

    def get_image_getter(self, task_name):
        def image_getter_helper(b0):
            mean_b0_img = nib.load(b0)
            mean_b0 = mean_b0_img.get_fdata()
            logger.warning((
                "It is reccomended that you provide a brain mask. "
                "It is provided with the brain_mask_definition argument. "
                "Otherwise, the default brain mask is calculated "
                "by using OTSU on the median-filtered B0 image. "
                "This can be unreliable. "))
            _, image_data = median_otsu(mean_b0, **self.median_otsu_kwargs)
            return nib.Nifti1Image(
                image_data.astype(np.float32),
                mean_b0_img.affine), dict(
                    source=b0,
                    technique="median_otsu applied to b0",
                    median_otsu_kwargs=self.median_otsu_kwargs)
        if task_name == "data" or task_name == "direct":
            return image_getter_helper
        else:
            return lambda data_imap: image_getter_helper(data_imap["b0"])

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        return self.get_image_getter("direct")(b0_file)


class LabelledImageFile(ImageFile, CombineImageMixin):
    """
    Define an image based on labels in a file.

    Parameters
    ----------
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    inclusive_labels : list of ints, optional
        The labels from the file to include from the boolean image.
        If None, no inclusive labels are applied.
    exclusive_labels : list of ints, optional
        The labels from the file to exclude from the boolean image.
        If None, no exclusive labels are applied.
        Default: None.
    combine : str, optional
        How to combine the boolean images generated by inclusive_labels
        and exclusive_labels. If "and", they will be and'd together.
        If "or", they will be or'd.
        Note: in this class, you will most likely want to either set
        inclusive_labels or exclusive_labels, not both,
        so combine will not matter.
        Default: "or"

    Examples
    --------
    brain_image_definition = LabelledImageFile(
        suffix="aseg",
        filters={"scope": "dmriprep"},
        exclusive_labels=[0])
    api.GroupAFQ(brain_image_definition=brain_image_definition)
    """

    def __init__(self, path=None, suffix=None, filters={},
                 inclusive_labels=None,
                 exclusive_labels=None, combine="or"):
        ImageFile.__init__(self, path, suffix, filters)
        CombineImageMixin.__init__(self, combine)
        self.inclusive_labels = inclusive_labels
        self.exclusive_labels = exclusive_labels

    # overrides ImageFile
    def apply_conditions(self, image_data_orig, image_file):
        # For different sets of labels, extract all the voxels that
        # have any / all of these values:
        self.reset_image_draft(image_data_orig.shape)
        if self.inclusive_labels is not None:
            for label in self.inclusive_labels:
                self.image_draft = self * (image_data_orig == label)
        if self.exclusive_labels is not None:
            for label in self.exclusive_labels:
                self.image_draft = self * (image_data_orig != label)

        meta = dict(source=image_file,
                    inclusive_labels=self.inclusive_labels,
                    exclusive_lavels=self.exclusive_labels,
                    combined_with=self.combine)
        return self.image_draft, meta


class ThresholdedImageFile(ImageFile, CombineImageMixin):
    """
    Define an image based on thresholding a file.
    Note that this should not be used to directly make a seed image
    or a stop image. In those cases, consider thresholding after
    interpolation, as in the example for ImageFile.

    Parameters
    ----------
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    lower_bound : float, optional
        Lower bound to generate boolean image from data in the file.
        If None, no lower bound is applied.
        Default: None.
    upper_bound : float, optional
        Upper bound to generate boolean image from data in the file.
        If None, no upper bound is applied.
        Default: None.
    combine : str, optional
        How to combine the boolean images generated by lower_bound
        and upper_bound. If "and", they will be and'd together.
        If "or", they will be or'd.
        Default: "and"

    Examples
    --------
    brain_image_definition = ThresholdedImageFile(
        suffix="BM",
        filters={"scope":"dmriprep"},
        lower_bound=0.1)
    api.GroupAFQ(brain_image_definition=brain_image_definition)
    """

    def __init__(self, path=None, suffix=None, filters={}, lower_bound=None,
                 upper_bound=None, combine="and"):
        ImageFile.__init__(self, path, suffix, filters)
        CombineImageMixin.__init__(self, combine)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    # overrides ImageFile
    def apply_conditions(self, image_data_orig, image_file):
        # Apply thresholds
        self.reset_image_draft(image_data_orig.shape)
        if self.upper_bound is not None:
            self.image_draft = self * (image_data_orig < self.upper_bound)
        if self.lower_bound is not None:
            self.image_draft = self * (image_data_orig > self.lower_bound)

        meta = dict(source=image_file,
                    upper_bound=self.upper_bound,
                    lower_bound=self.lower_bound,
                    combined_with=self.combine)
        return self.image_draft, meta


class ScalarImage(ImageDefinition):
    """
    Define an image based on a scalar.
    Does not apply any labels or thresholds;
    Generates image with floating point data.
    Useful for seed and stop images, where threshold can be applied
    after interpolation (see example).

    Parameters
    ----------
    scalar : str
        Scalar to threshold.
        Can be one of "dti_fa", "dti_md", "dki_fa", "dki_md".

    Examples
    --------
    seed_image = ScalarImage(
        "dti_fa")
    api.GroupAFQ(tracking_params={
        "seed_image": seed_image,
        "seed_threshold": 0.2})
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_name(self):
        return self.scalar

    def get_image_getter(self, task_name):
        def image_getter(data_imap):
            return nib.load(data_imap[self.scalar]), dict(
                FromScalar=self.scalar)
        return image_getter

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        if data_imap is not None:
            return self.get_image_getter("direct")(data_imap)
        else:
            raise ValueError((
                "ScalarImage cannot be used in this context, as they"
                "require later derivatives to be calculated"))


class ThresholdedScalarImage(ThresholdedImageFile, ScalarImage):
    """
    Define an image based on thresholding a scalar image.
    Note that this should not be used to directly make a seed image
    or a stop image. In those cases, consider thresholding after
    interpolation, as in the example for ScalarImage.

    Parameters
    ----------
    scalar : str
        Scalar to threshold.
        Can be one of "dti_fa", "dti_md", "dki_fa", "dki_md".
    lower_bound : float, optional
        Lower bound to generate boolean image from data in the file.
        If None, no lower bound is applied.
        Default: None.
    upper_bound : float, optional
        Upper bound to generate boolean image from data in the file.
        If None, no upper bound is applied.
        Default: None.
    combine : str, optional
        How to combine the boolean images generated by lower_bound
        and upper_bound. If "and", they will be and'd together.
        If "or", they will be or'd.
        Default: "and"

    Examples
    --------
    seed_image = ThresholdedScalarImage(
        "dti_fa",
        lower_bound=0.2)
    api.GroupAFQ(tracking_params={"seed_image": seed_image})
    """

    def __init__(self, scalar, lower_bound=None, upper_bound=None,
                 combine="and"):
        self.scalar = scalar
        CombineImageMixin.__init__(self, combine)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class PFTImage(ImageDefinition):
    """
    Define an image for use in PFT tractography. Only use
    if tracker set to 'pft' in tractography.

    Parameters
    ----------
    WM_probseg : ImageFile
        White matter segmentation file.
    GM_probseg : ImageFile
        Gray matter segmentation file.
    CSF_probseg : ImageFile
        Corticospinal fluid segmentation file.

    Examples
    --------
    stop_image = PFTImage(
        afm.ImageFile(suffix="WMprobseg"),
        afm.ImageFile(suffix="GMprobseg"),
        afm.ImageFile(suffix="CSFprobseg"))
    api.GroupAFQ(tracking_params={
        "stop_image": stop_image,
        "stop_threshold": "CMC",
        "tracker": "pft"})
    """

    def __init__(self, WM_probseg, GM_probseg, CSF_probseg):
        self.probsegs = (WM_probseg, GM_probseg, CSF_probseg)

    def find_path(self, bids_layout, from_path, subject, session):
        for probseg in self.probsegs:
            probseg.find_path(bids_layout, from_path, subject, session)

    def get_name(self):
        return "pft"

    def get_image_getter(self, task_name):
        return [probseg.get_image_getter(task_name)
                for probseg in self.probsegs]

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        raise ValueError("PFTImage cannot be used in this context")


class TemplateImage(ImageDefinition):
    """
    Define a scalar based on a template.
    This template will be transformed into subject space before use.

    Parameters
    ----------
    path : str
        path to the template.

    Examples
    --------
    my_scalar = TemplateImage(
        "path/to/my_scalar_in_MNI.nii.gz")
    api.GroupAFQ(scalars=["dti_fa", "dti_md", my_scalar])
    """

    def __init__(self, path):
        self.path = path

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def get_name(self):
        return name_from_path(self.path)

    def get_image_getter(self, task_name):
        def _image_getter_helper(mapping, reg_template):
            img = nib.load(self.path)
            img_data = resample(
                img.get_fdata(),
                reg_template,
                img.affine,
                reg_template.affine).get_fdata()

            scalar_data = mapping.transform_inverse(
                img_data, interpolation='nearest')
            return nib.Nifti1Image(
                scalar_data.astype(np.float32),
                reg_template.affine), dict(source=self.path)

        if task_name == "mapping":
            def image_getter(mapping, data_imap):
                return _image_getter_helper(
                    mapping, data_imap["reg_template"])
        else:
            def image_getter(mapping_imap, data_imap):
                return _image_getter_helper(
                    mapping_imap["mapping"], data_imap["reg_template"])
        return image_getter

    def get_image_direct(self, dwi, bids_info, b0_file, data_imap=None):
        raise ValueError((
            "ScalarImage cannot be used in this context, as they"
            "require later derivatives to be calculated"))
