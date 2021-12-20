import logging
from collections.abc import MutableMapping
import AFQ.data as afd


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
                 resample_to=None):
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
            Default: afd.read_mni_template()
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
            self.callosal_templates =\
                afd.read_callosum_templates(resample_to=self.resample_to)
        elif self.seg_algo.startswith("reco"):
            if self.seg_algo.endswith("80"):
                self.templates = afd.read_hcp_atlas(80)
            else:
                self.templates = afd.read_hcp_atlas(16)
        else:
            raise ValueError(
                "Input: %s is not a valid input`seg_algo`" % self.seg_algo)

    def gen(self, bundle_name):
        if not self.templates_loaded:
            self.load_templates()
        if self.seg_algo == "afq":
            if bundle_name in ["FA", "FP"]:
                bundle = {
                    'ROIs': [
                        self.templates[bundle_name + "_L"],
                        self.templates[bundle_name + "_R"],
                        self.callosal_templates["Callosum_midsag"]],
                    'rules': [True, True, True],
                    'prob_map': self.templates[bundle_name + "_prob_map"],
                    'cross_midline': True}
            elif bundle_name in CALLOSUM_BUNDLES:
                bundle = {
                    'ROIs': [
                        self.callosal_templates["L_" + bundle_name],
                        self.callosal_templates["R_" + bundle_name],
                        self.callosal_templates["Callosum_midsag"]],
                    'rules': [True, True, True],
                    'cross_midline': True}

            # SLF is a special case, because it has an exclusion ROI:
            elif bundle_name in ["SLF_L", "SLF_R"]:
                name = bundle_name[:-2]
                hemi = bundle_name[-2:]
                bundle = {
                    'ROIs': [
                        self.templates[name + '_roi1' + hemi],
                        self.templates[name + '_roi2' + hemi],
                        self.templates["SLFt_roi2" + hemi]],
                    'rules': [True, True, False],
                    'prob_map': self.templates[name + hemi + '_prob_map'],
                    'cross_midline': False}
            else:
                name = bundle_name[:-2]
                hemi = bundle_name[-2:]
                if (self.templates.get(name + '_roi1' + hemi)
                        and self.templates.get(name + '_roi2' + hemi)
                        and self.templates.get(name + hemi + '_prob_map')):
                    bundle = {
                        'ROIs': [
                            self.templates[name + '_roi1' + hemi],
                            self.templates[name + '_roi2' + hemi]],
                        'rules': [True, True],
                        'prob_map': self.templates[
                            name + hemi + '_prob_map'],
                        'cross_midline': False}
                else:
                    raise ValueError(f"{bundle_name} is not in AFQ templates")
            self._dict[bundle_name] = bundle
            self.resample_roi(bundle_name)
        elif self.seg_algo.startswith("reco"):
            self._dict[bundle_name] = self.templates[bundle_name]

    def gen_all(self):
        for bundle_name in self.bundle_names:
            if bundle_name not in self._dict:
                self.gen(bundle_name)

    def __setitem__(self, key, item):
        self._dict[key] = item
        self.bundle_names.append(key)
        self.resample_roi(key)

    def add_bundle_name(self, bundle_name):
        self.bundle_names.append(bundle_name)

    def __getitem__(self, key):
        if key not in self._dict:
            self.gen(key)
        return self._dict[key]

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
        return BundleDict(
            self._dict.copy(),
            seg_algo=self.seg_algo,
            resample_to=self.resample_to)

    def resample_roi(self, key):
        if self.resample_to:
            if "resampled" not in self._dict[key]\
                    or not self._dict[key]["resampled"]:
                for ii, roi in enumerate(self._dict[key]['ROIs']):
                    self._dict[key]['ROIs'][ii] =\
                        afd.read_resample_roi(
                            roi, resample_to=self.resample_to)
                self._dict[key]["resampled"] = True


class PediatricBundleDict(BundleDict):
    def __init__(self,
                 bundle_info=PEDIATRIC_BUNDLES,
                 seg_algo="afq",
                 resample_to=False):
        """
        Create a pediatric bundle dictionary, needed for the segmentation

        Parameters
        ----------
        bundle_info : list, optional
            A list of the pediatric bundles to be used in this case.
            Default: all of them

        seg_algo: only "afq" is supported
            The bundle segmentation algorithm to use.
                "afq" : Use waypoint ROIs + probability maps, as described
                in [Yeatman2012]_

        resample_to : Nifti1Image or bool, optional
            If set, templates will be resampled to the affine and shape of this
            image. If False, no resampling will be done.
            Default: False
        """
        BundleDict.__init__(self, bundle_info, seg_algo, resample_to)

    def gen_all(self):
        if self.all_gen:
            return
        if self.seg_algo == "afq":
            # Pediatric bundles differ from adult bundles:
            #   - A third ROI has been introduced for curvy tracts:
            #     ARC, ATR, CGC, IFO, and UCI
            #   - ILF posterior ROI has been split into two
            #     to separate ILF and mdLF
            #   - Addition of pAF and VOF ROIs
            #   - SLF ROIs are restricted to parietal cortex
            pediatric_templates = afd.read_pediatric_templates()

            # pediatric probability maps
            prob_map_order = [
                "ATR_L", "ATR_R", "CST_L", "CST_R", "CGC_L", "CGC_R",
                "HCC_L", "HCC_R", "FP", "FA", "IFO_L", "IFO_R", "ILF_L",
                "ILF_R", "SLF_L", "SLF_R", "UNC_L", "UNC_R",
                "ARC_L", "ARC_R", "MdLF_L", "MdLF_R"]

            prob_maps = pediatric_templates[
                'UNCNeo_JHU_tracts_prob-for-babyAFQ']
            prob_map_data = prob_maps.get_fdata()

            # pediatric bundle dict
            pediatric_bundles = {}

            # each bundles gets a digit identifier
            # (to be stored in the tractogram)
            uid = 1

            for name in PEDIATRIC_BUNDLES:
                # ROIs that cross the mid-line
                if name in ["FA", "FP"]:
                    pediatric_bundles[name] = {
                        'ROIs': [
                            pediatric_templates[name + "_L"],
                            pediatric_templates[name + "_R"],
                            pediatric_templates["mid-saggital"]],
                        'rules': [True, True, True],
                        'cross_midline': True,
                        'prob_map': prob_map_data[
                            ...,
                            prob_map_order.index(name)],
                        'uid': uid}
                    uid += 1
                # SLF is a special case, because it has an exclusion ROI:
                elif name == "SLF":
                    for hemi in ['_R', '_L']:
                        pediatric_bundles[name + hemi] = {
                            'ROIs': [
                                pediatric_templates[name + '_roi1' + hemi],
                                pediatric_templates[name + '_roi2' + hemi],
                                pediatric_templates["SLFt_roi2" + hemi]],
                            'rules': [True, True, False],
                            'cross_midline': False,
                            'prob_map': prob_map_data[
                                ...,
                                prob_map_order.index(name + hemi)],
                            'uid': uid}
                        uid += 1
                # Third ROI for curvy tracts
                elif name in ["ARC", "ATR", "CGC", "IFO", "UNC"]:
                    for hemi in ['_R', '_L']:
                        pediatric_bundles[name + hemi] = {
                            'ROIs': [
                                pediatric_templates[name + '_roi1' + hemi],
                                pediatric_templates[name + '_roi2' + hemi],
                                pediatric_templates[name + '_roi3' + hemi]],
                            'rules': [True, True, True],
                            'cross_midline': False,
                            'prob_map': prob_map_data[
                                ...,
                                prob_map_order.index(name + hemi)],
                            'uid': uid}
                        uid += 1
                elif name == "MdLF":
                    for hemi in ['_R', '_L']:
                        pediatric_bundles[name + hemi] = {
                            'ROIs': [
                                pediatric_templates[name + '_roi1' + hemi],
                                pediatric_templates[name + '_roi2' + hemi]],
                            'rules': [True, True],
                            'cross_midline': False,
                            # reuse probability map from ILF
                            'prob_map': prob_map_data[
                                ...,
                                prob_map_order.index("ILF" + hemi)],
                            'uid': uid}
                        uid += 1
                # Default: two ROIs within hemisphere
                else:
                    for hemi in ['_R', '_L']:
                        pediatric_bundles[name + hemi] = {
                            'ROIs': [
                                pediatric_templates[name + '_roi1' + hemi],
                                pediatric_templates[name + '_roi2' + hemi]],
                            'rules': [True, True],
                            'cross_midline': False,
                            'prob_map': prob_map_data[
                                ...,
                                prob_map_order.index(name + hemi)],
                            'uid': uid}
                        uid += 1
            self._dict = pediatric_bundles
        else:
            raise ValueError(
                "Input: %s is not a valid input`seg_algo`" % self.seg_algo)
        self.all_gen = True
