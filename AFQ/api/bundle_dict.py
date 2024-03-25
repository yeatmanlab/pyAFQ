import logging
from collections.abc import MutableMapping, Mapping

import AFQ.data.fetch as afd
import AFQ.utils.volume as auv
from AFQ.tasks.utils import get_fname, str_to_desc
from AFQ.definitions.utils import find_file

import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram

logging.basicConfig(level=logging.INFO)


__all__ = [
    "BundleDict",
    "default18_bd", "reco_bd",
    "callosal_bd", "cerebellar_bd",
    "baby_bd"]


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
# See: https://www.cmu.edu/dietrich/psychology/cognitiveaxon/documents/yeh_etal_2018.pdf  # noqa


DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


def default18_bd():
    templates = afd.read_templates(as_img=False)
    templates['ARC_roi1_L'] = templates['SLF_roi1_L']
    templates['ARC_roi1_R'] = templates['SLF_roi1_R']
    templates['ARC_roi2_L'] = templates['SLFt_roi2_L']
    templates['ARC_roi2_R'] = templates['SLFt_roi2_R']
    callosal_templates =\
        afd.read_callosum_templates(as_img=False)
    return BundleDict({
        'Left Anterior Thalamic': {
            'cross_midline': False,
            'include': [
                templates['ATR_roi1_L'],
                templates['ATR_roi2_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ATR_L_prob_map'],
            'start': templates['ATR_L_start'],
            'end': templates['ATR_L_end']},
        'Right Anterior Thalamic': {
            'cross_midline': False,
            'include': [templates['ATR_roi1_R'],
                        templates['ATR_roi2_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ATR_R_prob_map'],
            'start': templates['ATR_R_start'],
            'end': templates['ATR_R_end']},
        'Left Cingulum Cingulate': {
            'cross_midline': False,
            'include': [templates['CGC_roi2_L'],
                        templates['CGC_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CGC_L_prob_map'],
            'end': templates['CGC_L_start']},
        'Right Cingulum Cingulate': {
            'cross_midline': False,
            'include': [templates['CGC_roi2_R'],
                        templates['CGC_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CGC_R_prob_map'],
            'end': templates['CGC_R_start']},
        'Left Corticospinal': {
            'cross_midline': False,
            'include': [templates['CST_roi2_L'],
                        templates['CST_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CST_L_prob_map'],
            'end': templates['CST_L_start'],
            'start': templates['CST_L_end']},
        'Right Corticospinal': {
            'cross_midline': False,
            'include': [templates['CST_roi2_R'],
                        templates['CST_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CST_R_prob_map'],
            'end': templates['CST_R_start'],
            'start': templates['CST_R_end']},
        'Left Inferior Fronto-occipital': {
            'cross_midline': False,
            'include': [templates['IFO_roi2_L'],
                        templates['IFO_roi1_L']],
            'exclude': [templates['ARC_roi1_L']],
            'space': 'template',
            'prob_map': templates['IFO_L_prob_map'],
            'end': templates['IFO_L_start'],
            'start': templates['IFO_L_end']},
        'Right Inferior Fronto-occipital': {
            'cross_midline': False,
            'include': [templates['IFO_roi2_R'],
                        templates['IFO_roi1_R']],
            'exclude': [templates['ARC_roi1_R']],
            'space': 'template',
            'prob_map': templates['IFO_R_prob_map'],
            'end': templates['IFO_R_start'],
            'start': templates['IFO_R_end']},
        'Left Inferior Longitudinal': {
            'cross_midline': False,
            'include': [templates['ILF_roi2_L'],
                        templates['ILF_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ILF_L_prob_map'],
            'start': templates['ILF_L_end'],
            'end': templates['ILF_L_start']},
        'Right Inferior Longitudinal': {
            'cross_midline': False,
            'include': [templates['ILF_roi2_R'],
                        templates['ILF_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ILF_R_prob_map'],
            'start': templates['ILF_R_end'],
            'end': templates['ILF_R_start']},
        'Left Superior Longitudinal': {
            'cross_midline': False,
            'include': [templates['SLF_roi1_L'],
                        templates['SLF_roi2_L']],
            'exclude': [templates['SLFt_roi2_L']],
            'space': 'template',
            'prob_map': templates['SLF_L_prob_map'],
            'start': templates['SLF_L_start'],
            'end': templates['SLF_L_end']},
        'Right Superior Longitudinal': {
            'cross_midline': False,
            'include': [templates['SLF_roi1_R'],
                        templates['SLF_roi2_R']],
            'exclude': [templates['SLFt_roi2_R']],
            'space': 'template',
            'prob_map': templates['SLF_R_prob_map'],
            'start': templates['SLF_R_start'],
            'end': templates['SLF_R_end']},
        'Left Arcuate': {'cross_midline': False,
                         'include': [templates['SLF_roi1_L'],
                                     templates['SLFt_roi2_L']],
                         'exclude': [],
                         'space': 'template',
                         'prob_map': templates['ARC_L_prob_map'],
                         'start': templates['ARC_L_start'],
                         'end': templates['ARC_L_end']},
        'Right Arcuate': {'cross_midline': False,
                          'include': [templates['SLF_roi1_R'],
                                      templates['SLFt_roi2_R']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['ARC_R_prob_map'],
                          'start': templates['ARC_R_start'],
                          'end': templates['ARC_R_end']},
        'Left Uncinate': {'cross_midline': False,
                          'include': [templates['UNC_roi2_L'],
                                      templates['UNC_roi1_L']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['UNC_L_prob_map'],
                          'start': templates['UNC_L_end'],
                          'end': templates['UNC_L_start']},
        'Right Uncinate': {'cross_midline': False,
                           'include': [templates['UNC_roi2_R'],
                                       templates['UNC_roi1_R']],
                           'exclude': [],
                           'space': 'template',
                           'prob_map': templates['UNC_R_prob_map'],
                           'start': templates['UNC_R_end'],
                           'end': templates['UNC_R_start']},
        'Forceps Minor': {'cross_midline': True,
                          'include': [templates['FA_L'],
                                      callosal_templates['Callosum_midsag'],
                                      templates['FA_R']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['FA_prob_map'],
                          'start': templates['FA_start'],
                          'end': templates['FA_end']},
        'Forceps Major': {'cross_midline': True,
                          'include': [templates['FP_L'],
                                      callosal_templates['Callosum_midsag'],
                                      templates['FP_R']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['FP_prob_map'],
                          'start': templates['FP_start'],
                          'end': templates['FP_end']},
        'Left Posterior Arcuate': {'cross_midline': False,
                                   'include': [templates['SLFt_roi2_L']],
                                   'exclude': [templates['SLF_roi1_L']],
                                   'space': 'template',
                                   'start': templates['pARC_L_start'],
                                   'primary_axis': 2,
                                   'primary_axis_percentage': 40},
        'Right Posterior Arcuate': {'cross_midline': False,
                                    'include': [templates['SLFt_roi2_R']],
                                    'exclude': [templates['SLF_roi1_R']],
                                    'space': 'template',
                                    'start': templates['pARC_R_start'],
                                    'primary_axis': 2,
                                    'primary_axis_percentage': 40},
        'Left Vertical Occipital': {'cross_midline': False,
                                    'space': 'template',
                                    'start': templates['VOF_L_start'],
                                    'end': templates['VOF_L_end'],
                                    'inc_addtol': [4, 0],
                                    'primary_axis': 2,
                                    'primary_axis_percentage': 40},
        'Right Vertical Occipital': {'cross_midline': False,
                                     'space': 'template',
                                     'start': templates['VOF_R_start'],
                                     'end': templates['VOF_R_end'],
                                     'inc_addtol': [4, 0],
                                     'primary_axis': 2,
                                     'primary_axis_percentage': 40}})


def baby_bd():
    # Pediatric bundles differ from adult bundles:
    #   - A third ROI has been introduced for curvy tracts:
    #     ARC, ATR, CGC, IFO, and UCI
    #   - ILF posterior ROI has been split into two
    #     to separate ILF and mdLF
    #   - Addition of pAF and VOF ROIs
    #   - SLF ROIs are restricted to parietal cortex
    templates = afd.read_pediatric_templates()

    # pediatric probability maps
    prob_map_order = [
        "ATR_L", "ATR_R", "CST_L", "CST_R", "CGC_L", "CGC_R",
        "MdLF_L", "MdLF_R", "FP", "FA", "IFO_L", "IFO_R", "ILF_L",
        "ILF_R", "SLF_L", "SLF_R", "UNC_L", "UNC_R",
        "ARC_L", "ARC_R"]

    prob_maps = templates[
        'UNCNeo_JHU_tracts_prob-for-babyAFQ']
    prob_map_data = prob_maps.get_fdata()

    for bundle_name in prob_map_order:
        templates[bundle_name + "_prob_map"] = nib.Nifti1Image(
            prob_map_data[
                ...,
                prob_map_order.index(bundle_name)], prob_maps.affine)

    # reuse probability map from ILF
    templates["MdLF_L_prob_map"] = templates["ILF_L_prob_map"]
    templates["MdLF_R_prob_map"] = templates["ILF_R_prob_map"]
    return BundleDict({
        'Left Anterior Thalamic': {
            'cross_midline': False,
            'include': [
                templates['ATR_roi3_L'],
                templates['ATR_roi2_L'],
                templates['ATR_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ATR_L_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Right Anterior Thalamic': {'cross_midline': False,
                                    'include': [
                                        templates['ATR_roi3_R'],
                                        templates['ATR_roi2_R'],
                                        templates['ATR_roi1_R']],
                                    'exclude': [],
                                    'space': 'template',
                                    'prob_map': templates['ATR_R_prob_map'],
                                    'mahal': {'distance_threshold': 4}},
        'Left Cingulum Cingulate': {
            'cross_midline': False,
            'include': [templates['CGC_roi3_L'],
                        templates['CGC_roi2_L'],
                        templates['CGC_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CGC_L_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Right Cingulum Cingulate': {
            'cross_midline': False,
            'include': [templates['CGC_roi3_R'],
                        templates['CGC_roi2_R'],
                        templates['CGC_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CGC_R_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Left Corticospinal': {'cross_midline': False,
                               'include': [templates['CST_roi2_L'],
                                           templates['CST_roi1_L']],
                               'exclude': [],
                               'space': 'template',
                               'prob_map': templates['CST_L_prob_map'],
                               'mahal': {'distance_threshold': 4}},
        'Right Corticospinal': {'cross_midline': False,
                                'include': [templates['CST_roi2_R'],
                                            templates['CST_roi1_R']],
                                'exclude': [],
                                'space': 'template',
                                'prob_map': templates['CST_R_prob_map'],
                                'mahal': {'distance_threshold': 4}},
        'Left Inferior Fronto-occipital': {
            'cross_midline': False,
            'include': [templates['IFO_roi3_L'],
                        templates['IFO_roi2_L'],
                        templates['IFO_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['IFO_L_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Right Inferior Fronto-occipital': {
            'cross_midline': False,
            'include': [templates['IFO_roi3_R'],
                        templates['IFO_roi2_R'],
                        templates['IFO_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['IFO_R_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Left Inferior Longitudinal': {
            'cross_midline': False,
            'include': [templates['ILF_roi2_L'],
                        templates['ILF_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ILF_L_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Right Inferior Longitudinal': {
            'cross_midline': False,
            'include': [templates['ILF_roi2_R'],
                        templates['ILF_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['ILF_R_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Left Middle Longitudinal': {
            'cross_midline': False,
            'include': [templates['MdLF_roi2_L'],
                        templates['MdLF_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['MdLF_L_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Right Middle Longitudinal': {
            'cross_midline': False,
            'include': [templates['MdLF_roi2_R'],
                        templates['MdLF_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['MdLF_R_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Left Superior Longitudinal': {
            'cross_midline': False,
            'include': [templates['SLF_roi1_L'],
                        templates['SLF_roi2_L']],
            'exclude': [templates['SLFt_roi2_L']],
            'space': 'template',
            'prob_map': templates['SLF_L_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Right Superior Longitudinal': {
            'cross_midline': False,
            'include': [templates['SLF_roi1_R'],
                        templates['SLF_roi2_R']],
            'exclude': [templates['SLFt_roi2_R']],
            'space': 'template',
            'prob_map': templates['SLF_R_prob_map'],
            'mahal': {'distance_threshold': 4}},
        'Left Arcuate': {'cross_midline': False,
                         'include': [templates['ARC_roi1_L'],
                                     templates['ARC_roi2_L'],
                                     templates['ARC_roi3_L']],
                         'exclude': [],
                         'space': 'template',
                         'prob_map': templates['ARC_L_prob_map'],
                         'mahal': {'distance_threshold': 4}},
        'Right Arcuate': {'cross_midline': False,
                          'include': [templates['ARC_roi1_R'],
                                      templates['ARC_roi2_R'],
                                      templates['ARC_roi3_R']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['ARC_R_prob_map'],
                          'mahal': {'distance_threshold': 4}},
        'Left Uncinate': {'cross_midline': False,
                          'include': [templates['UNC_roi3_L'],
                                      templates['UNC_roi2_L'],
                                      templates['UNC_roi1_L']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['UNC_L_prob_map'],
                          'mahal': {'distance_threshold': 4}},
        'Right Uncinate': {'cross_midline': False,
                           'include': [templates['UNC_roi3_R'],
                                       templates['UNC_roi2_R'],
                                       templates['UNC_roi1_R']],
                           'exclude': [],
                           'space': 'template',
                           'prob_map': templates['UNC_R_prob_map'],
                           'mahal': {'distance_threshold': 4}},
        'Forceps Minor': {'cross_midline': True,
                          'include': [templates['FA_R'],
                                      templates['mid-saggital'],
                                      templates['FA_L']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['FA_prob_map'],
                          'mahal': {'distance_threshold': 4}},
        'Forceps Major': {'cross_midline': True,
                          'include': [templates['FP_R'],
                                      templates['mid-saggital'],
                                      templates['FP_L']],
                          'exclude': [],
                          'space': 'template',
                          'prob_map': templates['FP_prob_map'],
                          'mahal': {'distance_threshold': 4}},
        'Left Optic Radiation': {
            "include": [templates["OR_left_roi3"]],
            "start": templates["OR_leftThal"],
            "end": templates["OR_leftV1"],
            "cross_midline": False,
            "mahal": {"distance_threshold": 4}},
        'Right Optic Radiation': {
            "include": [templates["OR_right_roi3"]],
            "start": templates["OR_rightThal"],
            "end": templates["OR_rightV1"],
            "cross_midline": False,
            "mahal": {"distance_threshold": 4}},
        'Left Posterior Arcuate': {
            "include": [templates["SLFt_roi2_L"]],
            "exclude": [templates["SLF_roi1_L"]],
            "start": templates["pARC_L_start"],
            "end": templates["VOF_box_small_L"],
            "primary_axis": 2,
            "primary_axis_percentage": 40,
            "cross_midline": False,
            "mahal": {"distance_threshold": 4}},
        'Right Posterior Arcuate': {
            "include": [templates["SLFt_roi2_R"]],
            "exclude": [templates["SLF_roi1_R"]],
            "start": templates["pARC_R_start"],
            "end": templates["VOF_box_small_R"],
            "primary_axis": 2,
            "primary_axis_percentage": 40,
            "cross_midline": False,
            "mahal": {"distance_threshold": 4}},
        'Left Vertical Occipital': {
            "start": templates["VOF_L_start"],
            "end": templates["VOF_box_small_L"],
            "primary_axis": 2,
            "primary_axis_percentage": 40,
            "cross_midline": False,
            "mahal": {"distance_threshold": 4}},
        'Right Vertical Occipital': {
            "start": templates["VOF_R_start"],
            "end": templates["VOF_box_small_R"],
            "primary_axis": 2,
            "primary_axis_percentage": 40,
            "cross_midline": False,
            "mahal": {"distance_threshold": 4}}},
        seg_algo="afq",
        resample_to=afd.read_pediatric_templates()[
            'UNCNeo-withCerebellum-for-babyAFQ'])


def callosal_bd():
    callosal_templates =\
        afd.read_callosum_templates(as_img=False)
    return BundleDict({
        'Callosum Anterior Frontal': {
            'cross_midline': True,
            'include': [callosal_templates['R_AntFrontal'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_AntFrontal']],
            'exclude': [],
            'space': 'template'},
        'Callosum Motor': {
            'cross_midline': True,
            'include': [callosal_templates['R_Motor'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_Motor']],
            'exclude': [],
            'space': 'template'},
        'Callosum Occipital': {
            'cross_midline': True,
            'include': [callosal_templates['R_Occipital'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_Occipital']],
            'exclude': [],
            'space': 'template'},
        'Callosum Orbital': {
            'cross_midline': True,
            'include': [callosal_templates['R_Orbital'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_Orbital']],
            'exclude': [],
            'space': 'template'},
        'Callosum Posterior Parietal': {
            'cross_midline': True,
            'include': [callosal_templates['R_PostParietal'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_PostParietal']],
            'exclude': [],
            'space': 'template'},
        'Callosum Superior Frontal': {
            'cross_midline': True,
            'include': [callosal_templates['R_SupFrontal'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_SupFrontal']],
            'exclude': [],
            'space': 'template'},
        'Callosum Superior Parietal': {
            'cross_midline': True,
            'include': [callosal_templates['R_SupParietal'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_SupParietal']],
            'exclude': [],
            'space': 'template'},
        'Callosum Temporal': {
            'cross_midline': True,
            'include': [callosal_templates['R_Temporal'],
                        callosal_templates['Callosum_midsag'],
                        callosal_templates['L_Temporal']],
            'exclude': [],
            'space': 'template'}})


def reco_bd(n_bundles):
    """
    n_bundles: int
        Selects between 16 or 80 bundle atlas
    """
    templates = afd.read_hcp_atlas(n_bundles, as_file=True)
    return BundleDict(templates, seg_algo="reco")


def cerebellar_bd():
    cp_rois = afd.read_cp_templates()
    return BundleDict(
        {
            "Left Inferior Cerebellar Peduncle": {
                "include": [
                    cp_rois["ICP_L_inferior_prob"],
                    cp_rois["ICP_L_superior_prob"],
                ],
                "cross_midline": False,
            },
            "Right Inferior Cerebellar Peduncle": {
                "include": [
                    cp_rois["ICP_R_inferior_prob"],
                    cp_rois["ICP_R_superior_prob"],
                ],
                "cross_midline": False,
            },
            "Left Middle Cerebellar Peduncle": {
                "include": [
                    cp_rois["MCP_L_inferior_prob"],
                    cp_rois["MCP_R_superior_prob"],
                ],
                "exclude": [
                    cp_rois["SCP_L_inter_prob"],
                ],
                "cross_midline": True,
            },
            "Right Middle Cerebellar Peduncle": {
                "include": [
                    cp_rois["MCP_R_inferior_prob"],
                    cp_rois["MCP_L_superior_prob"],
                ],
                "exclude": [
                    cp_rois["SCP_R_inter_prob"],
                ],
                "cross_midline": True,
            },
            "Left Superior Cerebellar Peduncle": {
                "include": [
                    cp_rois["SCP_L_inferior_prob"],
                    cp_rois["SCP_L_inter_prob"],
                    cp_rois["SCP_R_superior_prob"],
                ],
                "exclude": [
                    cp_rois["SCP_L_superior_prob"],
                ],
                "cross_midline": True,
            },
            "Right Superior Cerebellar Peduncle": {
                "include": [
                    cp_rois["SCP_R_inferior_prob"],
                    cp_rois["SCP_R_inter_prob"],
                    cp_rois["SCP_L_superior_prob"],
                ],
                "exclude": [
                    cp_rois["SCP_R_superior_prob"],
                ],
                "cross_midline": True}})


def OR_bd():
    or_rois = afd.read_or_templates()

    return BundleDict({
        "Left Optic Radiation": {
            "include": [
                or_rois["left_OR_1"],
                or_rois["left_OR_2"]],
            "exclude": [
                or_rois["left_OP_MNI"],
                or_rois["left_TP_MNI"],
                or_rois["left_pos_thal_MNI"]],
            "start": or_rois['left_thal_MNI'],
            "end": or_rois['left_V1_MNI'],
            "cross_midline": False,
        },
        "Right Optic Radiation": {
            "include": [
                or_rois["right_OR_1"],
                or_rois["right_OR_2"]],
            "exclude": [
                or_rois["right_OP_MNI"],
                or_rois["right_TP_MNI"],
                or_rois["right_pos_thal_MNI"]],
            "start": or_rois['right_thal_MNI'],
            "end": or_rois['right_V1_MNI'],
            "cross_midline": False
        }
    })


class _BundleEntry(Mapping):
    """Describes how to recognize a single bundle, immutable"""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __setitem__(self, key, value):
        raise RuntimeError((
            "You cannot modify the properties of a bundle's definition. "
            "To modify a bundle's definition, replace that bundle's entry "
            "in the BundleDict."))


class BundleDict(MutableMapping):
    """
    Create a bundle dictionary, needed for the segmentation.

    Parameters
    ----------
    bundle_info : dict,
        A dictionary defining
        custom bundles. See `Defining Custom Bundle Dictionaries`
        in the `usage` section of pyAFQ's documentation for details.

    seg_algo: One of {"afq", "reco"}
        The bundle segmentation algorithm to use.
            "afq" : Use waypoint ROIs + probability maps, as described
            in [Yeatman2012]_
            "reco" : Use Recobundles [Garyfallidis2017]_

    resample_to : Nifti1Image or bool, optional
        If there are bundles in bundle_info with the 'space' attribute
        set to 'template', or with no 'space' attribute,
        their images (all ROIs and probability maps)
        will be resampled to the affine and shape of this image.
        If None, the MNI template will be used.
        If False, no resampling will be done.
        Default: None

    resample_subject_to : Nifti1Image or bool, optional
        If there are bundles in bundle_info with the 'space' attribute
        set to 'subject', their images (all ROIs and probability maps)
        will be resampled to the affine and shape of this image.
        If False, no resampling will be done.
        Default: None

    keep_in_memory : bool, optional
        Whether, once loaded, all ROIs and probability maps will stay
        loaded in memory within this object. By default, ROIs are loaded
        into memory on demand and no references to ROIs are kept, other
        than their paths. The default 18 bundles use ~6GB when all loaded.
        Default: False

    Examples
    --------
    # import OR ROIs and create a custom bundle dict
    # from them
    import AFQ.data.fetch as afd
    or_rois = afd.read_or_templates()

    bundles = BundleDict({
        "L_OR": {
            "include": [
                or_rois["left_OR_1"],  # these can be paths to Nifti files
                or_rois["left_OR_2"]],  # or they can Nifti images
            "exclude": [
                or_rois["left_OP_MNI"],
                or_rois["left_TP_MNI"],
                or_rois["left_pos_thal_MNI"]],
            "start": or_rois['left_thal_MNI'],
            "end": or_rois['left_V1_MNI'],
            "cross_midline": False,
        },
        "R_OR": {
            "include": [
                or_rois["right_OR_1"],
                or_rois["right_OR_2"]],
            "exclude": [
                or_rois["right_OP_MNI"],
                or_rois["right_TP_MNI"],
                or_rois["right_pos_thal_MNI"]],
            "start": or_rois['right_thal_MNI'],
            "end": or_rois['right_V1_MNI'],
            "cross_midline": False
        }
    })
    """

    def __init__(self,
                 bundle_info,
                 seg_algo="afq",
                 resample_to=None,
                 resample_subject_to=False,
                 keep_in_memory=False):
        if not (isinstance(bundle_info, dict)):
            raise TypeError((
                f"bundle_info must be a dict,"
                f" currently a {type(bundle_info)}"))
        self.seg_algo = seg_algo.lower()
        if resample_to is None:
            resample_to = afd.read_mni_template()
        self.resample_to = resample_to
        self.resample_subject_to = resample_subject_to
        self.keep_in_memory = keep_in_memory
        self.has_bids_info = False
        self.max_includes = 3

        self._dict = {}
        self.bundle_names = []
        for key, item in bundle_info.items():
            self.__setitem__(key, item)

        self.logger = logging.getLogger('AFQ')
        if self.seg_algo == "afq":
            if "Forceps Major" in self.bundle_names\
                    and "Callosum Occipital" in self.bundle_names:
                self.logger.info((
                    "Forceps Major and Callosum Occipital bundles"
                    " are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Callosum Occipital will be used."))
                self.bundle_names.remove("Forceps Major")
            if "Forceps Minor" in self.bundle_names\
                    and "Callosum Orbital" in self.bundle_names:
                self.logger.info((
                    "Forceps Minor and Callosum Orbital bundles"
                    " are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Callosum Orbital will be used."))
                self.bundle_names.remove("Forceps Minor")
            if "Forceps Minor" in self.bundle_names\
                    and "Callosum Anterior Frontal" in self.bundle_names:
                self.logger.info((
                    "Forceps Minor and Callosum Anterior Frontal bundles"
                    " are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Callosum Anterior Frontal will be used."))
                self.bundle_names.remove("Forceps Minor")

    def update_max_includes(self, new_max):
        if new_max > self.max_includes:
            self.max_includes = new_max

    def set_bids_info(self, bids_layout, bids_path, subject, session):
        """
        Provide the bids_layout, a nearest path,
        and the subject and session information
        to load ROIS from BIDS
        """
        self.has_bids_info = True
        self._bids_info = bids_layout
        self._bids_path = bids_path
        self._subject = subject
        self._session = session

    def _cond_load(self, roi_or_sl, resample_to):
        """
        Load ROI or streamline if not already loaded
        """
        if isinstance(roi_or_sl, dict):
            if not self.has_bids_info:
                raise ValueError((
                    "Attempted to load an ROI using BIDS description without "
                    "First providing BIDS information."))
            suffix = roi_or_sl.get("suffix", "dwi")
            roi_or_sl = find_file(
                self._bids_info, self._bids_path,
                roi_or_sl,
                suffix,
                self._session, self._subject)
        if isinstance(roi_or_sl, str):
            if self.seg_algo == "afq":
                return afd.read_resample_roi(
                    roi_or_sl,
                    resample_to=resample_to)
            elif self.seg_algo.startswith("reco"):
                return load_tractogram(
                    roi_or_sl,
                    'same',
                    bbox_valid_check=False).streamlines
        else:
            return roi_or_sl

    def get_b_info(self, b_name):
        return self._dict[b_name]

    def __getitem__(self, key):
        if isinstance(key, tuple) or isinstance(key, list):
            # Generates a copy of this BundleDict with only the bundle names
            # from the tuple/list
            new_bd = {}
            for b_name in key:
                if b_name in self._dict:
                    new_bd[b_name] = self._dict[b_name]
                else:
                    raise ValueError(f"{b_name} is not in this BundleDict")

            return self.__class__(
                new_bd,
                seg_algo=self.seg_algo,
                resample_to=self.resample_to,
                resample_subject_to=self.resample_subject_to,
                keep_in_memory=self.keep_in_memory)
        else:
            if not self.keep_in_memory:
                _item = self._dict[key].copy()
                _res = self._cond_load_bundle(key, dry_run=True)
                if _res is not None:
                    _item.update(_res)
                _item = _BundleEntry(_item)
            else:
                if "loaded" not in self._dict[key] or\
                        not self._dict[key]["loaded"]:
                    self._cond_load_bundle(key)
                    self._dict[key]["loaded"] = True
                if "resampled" not in self._dict[key] or not self._dict[
                        key]["resampled"]:
                    self._resample_roi(key)
                _item = _BundleEntry(self._dict[key].copy())
            return _item

    def __setitem__(self, key, item):
        self._dict[key] = item
        if hasattr(item, "get"):
            self.update_max_includes(len(item.get("include", [])))
        if key not in self.bundle_names:
            self.bundle_names.append(key)

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
        return iter(self._dict)

    def copy(self):
        """
        Generates a copy of this BundleDict where the internal dictionary
        is a copy of this BundleDict's internal dictionary.
        Useful if you want to add or remove bundles from a copy
        of a BundleDict.

        Returns
        ---------
        bundle_dict : BundleDict
            Euclidean norms of vectors.
        """
        return self.__class__(
            self._dict.copy(),
            seg_algo=self.seg_algo,
            resample_to=self.resample_to,
            resample_subject_to=self.resample_subject_to,
            keep_in_memory=self.keep_in_memory)

    def apply_to_rois(self, b_name, func, *args,
                      dry_run=False,
                      **kwargs):
        """
        Applies some transformation to all ROIs (include, exclude, end, start)
        and the prob_map in a given bundle.

        Parameters
        ----------
        b_name : name
            bundle name of bundle whose ROIs will be transformed.
        func : function
            function whose first argument must be a Nifti1Image and which
            returns a Nifti1Image
        dry_run : bool
            Whether to actually apply changes returned by `func` to the ROIs.
            If has_return is False, dry_run is not used.
        *args :
            Additional arguments for func
        **kwargs
            Optional arguments for func

        Returns
        -------
        A dictionary where keys are
        the roi type and values are the transformed ROIs.
        """
        return_vals = {}
        if self.seg_algo == "afq":
            for roi_type in ["include", "exclude", "start", "end", "prob_map"]:
                if roi_type in self._dict[b_name]:
                    if roi_type in ["start", "end", "prob_map"]:
                        return_vals[roi_type] = func(
                            self._dict[b_name][roi_type], *args, **kwargs)
                    else:
                        changed_rois = []
                        for _roi in self._dict[b_name][roi_type]:
                            changed_rois.append(func(
                                _roi, *args, **kwargs))
                        return_vals[roi_type] = changed_rois
        elif self.seg_algo.startswith("reco"):
            if b_name == "whole_brain":
                return_vals = func(
                    self._dict[b_name], *args, **kwargs)
            else:
                for sl_type in ["sl", "centroid"]:
                    return_vals[sl_type] = func(
                        self._dict[b_name][sl_type],
                        *args, **kwargs)
        if not dry_run:
            for roi_type, roi in return_vals.items():
                self._dict[b_name][roi_type] = roi
        return return_vals

    def _cond_load_bundle(self, b_name, dry_run=False):
        """
        Given a bundle name, resample all ROIs and prob maps
        into either template or subject space for that bundle,
        depending on its "space" attribute.

        Parameters
        ----------
        b_name : str
            Name of the bundle to be resampled.
        """
        if self.seg_algo == "afq":
            if self.is_bundle_in_template(b_name):
                resample_to = self.resample_to
            else:
                resample_to = self.resample_subject_to
        else:
            resample_to = None
        return self.apply_to_rois(
            b_name,
            self._cond_load,
            resample_to,
            dry_run=dry_run)

    def is_bundle_in_template(self, bundle_name):
        return "space" not in self._dict[bundle_name]\
            or self._dict[bundle_name]["space"] == "template"

    def _roi_transform_helper(self, roi, mapping, new_affine, bundle_name):
        roi = afd.read_resample_roi(roi, self.resample_to)
        warped_img = auv.transform_inverse_roi(
            roi.get_fdata(),
            mapping,
            bundle_name=bundle_name)
        warped_img = nib.Nifti1Image(warped_img, new_affine)
        return warped_img

    def transform_rois(self, bundle_name, mapping, new_affine,
                       base_fname=None):
        """
        Get the bundle definition with transformed ROIs
        for a given bundle into a
        given subject space using a given mapping.
        Will only run on bundles which are in template
        space, otherwise will just return the bundle
        definition without transformation.

        Parameters
        ----------
        bundle_name : str
            Name of the bundle to be transformed.
        mapping : DiffeomorphicMap object
            A mapping between DWI space and a template.
        new_affine : array
            Affine of space transformed into.
        base_fname : str, optional
            Base file path to save ROIs too. Additional BIDS
            descriptors will be added to this file path. If None,
            do not save the ROIs.

        Returns
        -------
        If base_fname is None, a dictionary where keys are
        the roi type and values are the transformed ROIs.
        Otherwise, a list of file names where the transformed
        ROIs are saved.
        """
        if self.is_bundle_in_template(bundle_name):
            transformed_rois = self.apply_to_rois(
                bundle_name,
                self._roi_transform_helper,
                mapping,
                new_affine,
                bundle_name,
                dry_run=True)
        else:
            transformed_rois = self.apply_to_rois(
                bundle_name,
                self._cond_load,
                self.resample_subject_to,
                dry_run=True)

        if base_fname is not None:
            fnames = []
            for roi_type, rois in transformed_rois.items():
                if not isinstance(rois, list):
                    rois = [rois]
                for ii, roi in enumerate(rois):
                    fname = get_fname(
                        base_fname,
                        "_space-subject_desc-"
                        f"{str_to_desc(bundle_name)}{roi_type}{ii}"
                        "_mask.nii.gz")
                    nib.save(
                        nib.Nifti1Image(
                            roi.get_fdata().astype(np.float32),
                            roi.affine), fname)
                    fnames.append(fname)
            return fnames
        else:
            return transformed_rois

    def __add__(self, other):
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
                if not np.allclose(
                        getattr(self, resample).affine,
                        getattr(other, resample).affine):
                    raise ValueError((
                        f"Adding BundleDicts where {resample} affines"
                        f" do not match. {resample} affines are"
                        f"{getattr(self, resample).affine} and "
                        f"{getattr(other, resample).affine}"))
                if not np.allclose(
                        getattr(self, resample).header['dim'],
                        getattr(other, resample).header['dim']):
                    raise ValueError((
                        f"Adding BundleDicts where {resample} dimensions"
                        f" do not match. {resample} dimensions are"
                        f"{getattr(self, resample).header['dim']} and "
                        f"{getattr(other, resample).header['dim']}"))
        return self.__class__(
            {**self._dict, **other._dict},
            self.seg_algo,
            self.resample_to,
            self.resample_subject_to,
            self.keep_in_memory)
