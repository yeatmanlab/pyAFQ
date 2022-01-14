import logging
from collections.abc import MutableMapping
import AFQ.data.fetch as afd
import numpy as np


logging.basicConfig(level=logging.INFO)


__all__ = ["PediatricBundleDict", "BundleDict"]


def do_preprocessing():
    raise NotImplementedError


def append_l_r(bundle_list, no_lr_list):
    new_bundle_list = []
    for bundle in bundle_list:
        if bundle in no_lr_list:
            new_bundle_list.append(bundle)
        else:
            new_bundle_list.append(bundle + "_L")
            new_bundle_list.append(bundle + "_R")
    return new_bundle_list


BUNDLES = ["ATR", "CGC", "CST", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]
BUNDLES = append_l_r(BUNDLES, ["FA", "FP"])

CALLOSUM_BUNDLES = ["AntFrontal", "Motor", "Occipital", "Orbital",
                    "PostParietal", "SupFrontal", "SupParietal",
                    "Temporal"]

# See: https://www.cmu.edu/dietrich/psychology/cognitiveaxon/documents/yeh_etal_2018.pdf  # noqa

RECO_UNIQUE = [
    'CCMid', 'CC_ForcepsMajor', 'CC_ForcepsMinor', 'MCP', 'AC', 'PC', 'SCP',
    'V', 'CC', 'F_L_R']

RECO_BUNDLES_16 = [
    'CST', 'C', 'F', 'UF', 'MCP', 'AF', 'CCMid',
    'CC_ForcepsMajor', 'CC_ForcepsMinor', 'IFOF']
RECO_BUNDLES_16 = append_l_r(RECO_BUNDLES_16, RECO_UNIQUE)

RECO_BUNDLES_80 = ["AC", "AF", "AR", "AST", "C", "CB", "CC_ForcepsMajor",
                   "CC_ForcepsMinor", "CC", "CCMid", "CNII", "CNIII",
                   "CNIV", "CNV", "CNVII", "CNVIII", "CS", "CST", "CT",
                   "CTT", "DLF", "EMC", "F_L_R", "FPT", "ICP", "IFOF", "ILF",
                   "LL", "MCP", "MdLF", "ML", "MLF", "OPT", "OR", "PC", "PPT",
                   "RST", "SCP", "SLF", "STT", "TPT", "UF", "V", "VOF"]
RECO_BUNDLES_80 = append_l_r(RECO_BUNDLES_80, RECO_UNIQUE)

PEDIATRIC_BUNDLES = [
    "ARC", "ATR", "CGC", "CST", "FA", "FP", "IFO", "ILF", "MdLF", "SLF", "UNC"]
PEDIATRIC_BUNDLES = append_l_r(PEDIATRIC_BUNDLES, ["FA", "FP"])

DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


class BundleDict(MutableMapping):
    def __init__(self,
                 bundle_info=BUNDLES,
                 seg_algo="afq",
                 resample_to=None,
                 resample_subject_to=None):
        """
        Create a bundle dictionary, needed for the segmentation

        Parameters
        ----------
        bundle_info : list or dict, optional
            A list of the bundles to be used, or a dictionary defining
            custom bundles.
            Default: AFQ.api.bundle_dict.BUNDLES

        seg_algo: One of {"afq", "reco", "reco16", "reco80"}
            The bundle segmentation algorithm to use.
                "afq" : Use waypoint ROIs + probability maps, as described
                in [Yeatman2012]_
                "reco" / "reco16" : Use Recobundles [Garyfallidis2017]_
                with a 16-bundle set.
                "reco80": Use Recobundles with an 80-bundle set.

        resample_to : Nifti1Image or bool, optional
            If set, templates will be resampled to the affine and shape of this
            image. If None, the MNI template will be used.
            If False, no resampling will be done.
            Default: None

        resample_subject_to : Nifti1Image or bool, optional
            If there are ROIs with the 'space' attribute
            set to 'subject', those ROIs will be resampled to the affine
            and shape of this image.
            If None, the template will be overriden when passed to
            an API class.
            If False, no resampling will be done.
            Default: None
        """
        if not (isinstance(bundle_info, dict)
                or isinstance(bundle_info, list)):
            raise TypeError((
                f"bundle_info must be a dict or a list,"
                f" currently a {type(bundle_info)}"))
        self.seg_algo = seg_algo.lower()
        if resample_to is None:
            resample_to = afd.read_mni_template()
        self.resample_to = resample_to
        self.resample_subject_to = resample_subject_to

        self._dict = {}
        self.bundle_names = []
        self.templates_loaded = False
        if isinstance(bundle_info, dict):
            for key, item in bundle_info.items():
                self.__setitem__(key, item)
        else:
            for bundle_name in bundle_info:
                self.add_bundle_name(bundle_name)

        self.logger = logging.getLogger('AFQ.api')

        if self.seg_algo == "afq":
            if "FP" in self.bundle_names\
                    and "Occipital" in self.bundle_names:
                self.logger.warning((
                    "FP and Occipital bundles are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Occipital will be used."))
                self.bundle_names.remove("FP")
            if "FA" in self.bundle_names\
                    and "Orbital" in self.bundle_names:
                self.logger.warning((
                    "FA and Orbital bundles are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Orbital will be used."))
                self.bundle_names.remove("FA")
            if "FA" in self.bundle_names\
                    and "AntFrontal" in self.bundle_names:
                self.logger.warning((
                    "FA and AntFrontal bundles are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only AntFrontal will be used."))
                self.bundle_names.remove("FA")

    def load_templates(self):
        if self.seg_algo == "afq":
            self.templates =\
                afd.read_templates(resample_to=self.resample_to)
            # For the arcuate, we need to rename a few of these
            # and duplicate the SLF ROI:
            self.templates['ARC_roi1_L'] = self.templates['SLF_roi1_L']
            self.templates['ARC_roi1_R'] = self.templates['SLF_roi1_R']
            self.templates['ARC_roi2_L'] = self.templates['SLFt_roi2_L']
            self.templates['ARC_roi2_R'] = self.templates['SLFt_roi2_R']
            callosal_templates =\
                afd.read_callosum_templates(resample_to=self.resample_to)
            endpoint_templates =\
                afd.bundles_to_aal(self.bundle_names)
            self.templates = {
                **self.templates,
                **callosal_templates,
                **endpoint_templates}
        elif self.seg_algo.startswith("reco"):
            if self.seg_algo.endswith("80"):
                self.templates = afd.read_hcp_atlas(80)
            else:
                self.templates = afd.read_hcp_atlas(16)
        else:
            raise ValueError(
                "Input: %s is not a valid input`seg_algo`" % self.seg_algo)
        self.templates_loaded = True

    def gen(self, bundle_name):
        if not self.templates_loaded:
            self.load_templates()
        if self.seg_algo == "afq":
            name = bundle_name[:-2]
            hemi = bundle_name[-2:]
            if bundle_name in CALLOSUM_BUNDLES:
                roi_name1 = "L_" + bundle_name
                roi_name2 = "R_" + bundle_name
            elif bundle_name in ["FA", "FP"]:
                roi_name1 = bundle_name + "_L"
                roi_name2 = bundle_name + "_R"
            else:
                roi_name1 = name + '_roi1' + hemi
                roi_name2 = name + '_roi2' + hemi
            if (roi_name1 in self.templates
                    and roi_name2 in self.templates):
                roi_dict = {}
                roi_dict['cross_midline'] = False
                roi_dict['include'] = [
                    self.templates[roi_name1],
                    self.templates[roi_name2]]
                roi_dict['exclude'] = []
                roi_dict['space'] = 'template'
                if name + '_roi3' + hemi in self.templates:
                    roi_dict['include'].append(
                        self.templates[name + '_roi3' + hemi])
                if name == "SLF":
                    roi_dict['exclude'].append(
                        self.templates["SLFt_roi2" + hemi])
                if bundle_name in CALLOSUM_BUNDLES\
                        or bundle_name in ["FA", "FP"]:
                    roi_dict['include'].append(
                        self.templates["Callosum_midsag"])
                    roi_dict['cross_midline'] = True
                if bundle_name + '_prob_map' in self.templates:
                    roi_dict['prob_map'] = self.templates[
                        bundle_name + '_prob_map']
                if bundle_name + "_start" in self.templates and self.templates[
                        bundle_name + "_start"] is not None:
                    roi_dict['start'] = self.templates[
                        bundle_name + "_start"]
                if bundle_name + "_end" in self.templates and self.templates[
                        bundle_name + "_end"] is not None:
                    roi_dict['end'] = self.templates[
                        bundle_name + "_end"]
                self._dict[bundle_name] = roi_dict
                self.resample_roi(bundle_name)
            else:
                raise ValueError(f"{bundle_name} is not in AFQ templates")
        elif self.seg_algo.startswith("reco"):
            self._dict[bundle_name] = self.templates[bundle_name]

    def gen_all(self):
        for bundle_name in self.bundle_names:
            if bundle_name not in self._dict:
                self.gen(bundle_name)

    def add_bundle_name(self, bundle_name):
        self.bundle_names.append(bundle_name)

    def __getitem__(self, key):
        if key not in self._dict and key in self.bundle_names:
            self.gen(key)
        if self.resample_to and key in self._dict and (
            "resampled" not in self._dict[key] or not self._dict[
                key]["resampled"]):
            self.resample_roi(key)
        return self._dict[key]

    def __setitem__(self, key, item):
        self._dict[key] = item
        self.bundle_names.append(key)
        self.resample_roi(key)

    def __len__(self):
        return len(self.bundle_names)

    def __delitem__(self, key):
        if key not in self._dict and key not in self.bundle_names:
            raise KeyError(f"{key} not found")
        if key in self._dict:
            del self._dict[key]
        else:
            raise RuntimeError((
                f"{key} not found in internal dictionary, "
                f"but found in bundle_names"))
        if key in self.bundle_names:
            self.bundle_names.remove(key)
        else:
            raise RuntimeError((
                f"{key} not found in bundle_names, "
                f"but found in internal dictionary"))

    def __iter__(self):
        self.gen_all()
        return iter(self._dict)

    def copy(self):
        self.gen_all()
        return self.__class__(
            self._dict.copy(),
            seg_algo=self.seg_algo,
            resample_to=self.resample_to)

    def apply_to_rois(self, b_name, func, *args, **kwargs):
        for roi_type in ["include", "exclude", "start", "end"]:
            if roi_type in self._dict[b_name]:
                roi = self._dict[b_name][roi_type]
                if roi_type in ["start", "end"]:
                    rois = [roi]
                else:
                    rois = roi
                changed_rois = []
                for roi in rois:
                    changed_rois.append(func(roi, *args, **kwargs))
                if roi_type in ["start", "end"]:
                    self._dict[b_name][roi_type] = changed_rois[0]
                else:
                    self._dict[b_name][roi_type] = changed_rois

    def resample_roi(self, b_name):
        if self.resample_to:
            if "resampled" not in self._dict[b_name]\
                    or not self._dict[b_name]["resampled"]:
                if "space" not in self._dict[b_name]\
                        or self._dict[b_name]["space"] == "template":
                    resample_to = self.resample_to
                else:
                    resample_to = self.resample_subject_to
                try:
                    self.apply_to_rois(
                        b_name,
                        afd.read_resample_roi,
                        resample_to=resample_to)
                    self._dict[b_name]["resampled"] = True
                except AttributeError as e:
                    if "'ScalarFile' object" in str(e):
                        self._dict[b_name]["resampled"] = False
                    else:
                        raise

    def __add__(self, other):
        self.gen_all()
        other.gen_all()
        if self.seg_algo != other.seg_algo:
            raise ValueError((
                "Adding BundleDicts where seg_algo do not match."
                f"seg_algo's are {self.seg_algo} and {other.seg_algo}"))
        for resample in ["resample_to", "resample_subject_to"]:
            if not getattr(self, resample)\
                    or not getattr(other, resample)\
                    or getattr(self, resample) is None\
                    or getattr(other, resample) is None:
                if getattr(self, resample) != getattr(other, resample):
                    raise ValueError((
                        f"Adding BundleDicts where {resample} do not match."
                        f"{resample}'s are {getattr(self, resample)} and "
                        f"{getattr(other, resample)}"))
            else:
                if not np.all(
                    getattr(self, resample).affine == getattr(
                        other, resample).affine):
                    raise ValueError((
                        f"Adding BundleDicts where {resample} affines"
                        f" do not match. {resample} affines are"
                        f"{getattr(self, resample).affine} and "
                        f"{getattr(other, resample).affine}"))
                if not np.all(
                    getattr(self, resample).header['dim']
                        == getattr(other, resample).header['dim']):
                    raise ValueError((
                        f"Adding BundleDicts where {resample} dimensions"
                        f" do not match. {resample} dimensions are"
                        f"{getattr(self, resample).header['dim']} and "
                        f"{getattr(other, resample).header['dim']}"))
        return self.__class__(
            {**self._dict, **other._dict},
            self.seg_algo,
            self.resample_to)


class PediatricBundleDict(BundleDict):
    def __init__(self,
                 bundle_info=PEDIATRIC_BUNDLES,
                 seg_algo="afq",
                 resample_to=None,
                 resample_subject_to=None):
        """
        Create a pediatric bundle dictionary, needed for the segmentation

        Parameters
        ----------
        bundle_info : list or dict, optional
            A list of the bundles to be used, or a dictionary defining
            custom bundles.
            Default: AFQ.api.bundle_dict.PEDIATRIC_BUNDLES

        seg_algo: only "afq" is supported
            The bundle segmentation algorithm to use.
                "afq" : Use waypoint ROIs + probability maps, as described
                in [Yeatman2012]_

        resample_to : Nifti1Image or bool, optional
            If set, templates will be resampled to the affine and shape of this
            image. If None, this will be used:
            afd.read_pediatric_templates()['UNCNeo-withCerebellum-for-babyAFQ']
            If False, no resampling will be done.
            Default: None

        resample_subject_to : Nifti1Image or bool, optional
            If there are ROIs with the 'space' attribute
            set to 'subject', those ROIs will be resampled to the affine
            and shape of this image.
            If None, the template will be overriden when passed to
            an API class.
            If False, no resampling will be done.
            Default: None
        """
        if resample_to is None:
            resample_to = afd.read_pediatric_templates()[
                'UNCNeo-withCerebellum-for-babyAFQ']
        self.resample_to = resample_to
        BundleDict.__init__(
            self, bundle_info, seg_algo,
            resample_to, resample_subject_to)

    def load_templates(self):
        # Pediatric bundles differ from adult bundles:
        #   - A third ROI has been introduced for curvy tracts:
        #     ARC, ATR, CGC, IFO, and UCI
        #   - ILF posterior ROI has been split into two
        #     to separate ILF and mdLF
        #   - Addition of pAF and VOF ROIs
        #   - SLF ROIs are restricted to parietal cortex
        self.templates = afd.read_pediatric_templates()

        # pediatric probability maps
        prob_map_order = [
            "ATR_L", "ATR_R", "CST_L", "CST_R", "CGC_L", "CGC_R",
            "HCC_L", "HCC_R", "FP", "FA", "IFO_L", "IFO_R", "ILF_L",
            "ILF_R", "SLF_L", "SLF_R", "UNC_L", "UNC_R",
            "ARC_L", "ARC_R", "MdLF_L", "MdLF_R"]

        prob_maps = self.pediatric_templates[
            'UNCNeo_JHU_tracts_prob-for-babyAFQ']
        prob_map_data = prob_maps.get_fdata()

        self.templates["Callosum_midsag"] = self.templates["mid-saggital"]

        for bundle_name in PEDIATRIC_BUNDLES:
            self.templates[bundle_name + "_prob_map"] = prob_map_data[
                ...,
                prob_map_order.index(bundle_name)]

        # reuse probability map from ILF
        self.templates["MdLF_L_prob_map"] = self.templates["ILF_L_prob_map"]
        self.templates["MdLF_R_prob_map"] = self.templates["ILF_R_prob_map"]
