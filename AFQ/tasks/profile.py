import nibabel as nib
import numpy as np
import pandas as pd

import pimms

from AFQ.tasks.utils import as_file
from AFQ.utils.bin import get_default_args

from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.tracking.streamline import set_number_of_points, values_from_volume


@pimms.calc("profiles_file")
@as_file('_profiles.csv', include_track=True, include_seg=True)
def tract_profiles(subses_dict, clean_bundles_file, bundle_dict,
                   scalar_dict, profile_weights, dwi_affine,
                   tracking_params, segmentation_params):
    keys = []
    vals = []
    for k in bundle_dict.keys():
        if k != "whole_brain":
            keys.append(bundle_dict[k]['uid'])
            vals.append(k)
    reverse_dict = dict(zip(keys, vals))

    bundle_names = []
    node_numbers = []
    profiles = np.empty((len(scalar_dict), 0)).tolist()
    this_profile = np.zeros((len(scalar_dict), 100))

    trk = nib.streamlines.load(clean_bundles_file)
    for b in np.unique(
            trk.tractogram.data_per_streamline['bundle']):
        idx = np.where(
            trk.tractogram.data_per_streamline['bundle'] == b)[0]
        this_sl = trk.streamlines[idx]
        bundle_name = reverse_dict[b]
        for ii, (scalar, scalar_file) in enumerate(scalar_dict.items()):
            scalar_data = nib.load(scalar_file).get_fdata()
            if isinstance(profile_weights, str):
                if profile_weights == "gauss":
                    this_prof_weights = gaussian_weights(this_sl)
                elif profile_weights == "median":
                    # weights bundle to only return the mean
                    def _median_weight(bundle):
                        fgarray = set_number_of_points(bundle, 100)
                        values = np.array(
                            values_from_volume(
                                scalar_data,
                                fgarray,
                                dwi_affine))
                        weights = np.zeros(values.shape)
                        for ii, jj in enumerate(
                            np.argsort(values, axis=0)[
                                len(values) // 2, :]):
                            weights[jj, ii] = 1
                        return weights
                    this_prof_weights = _median_weight
            else:
                this_prof_weights = profile_weights
            this_profile[ii] = afq_profile(
                scalar_data,
                this_sl,
                dwi_affine,
                weights=this_prof_weights)
            profiles[ii].extend(list(this_profile[ii]))
        nodes = list(np.arange(this_profile[0].shape[0]))
        bundle_names.extend([bundle_name] * len(nodes))
        node_numbers.extend(nodes)

    profile_dict = dict()
    profile_dict["tractID"] = bundle_names
    profile_dict["nodeID"] = node_numbers
    for ii, scalar in enumerate(scalar_dict.keys()):
        profile_dict[scalar] = profiles[ii]

    profile_dframe = pd.DataFrame(profile_dict)
    meta = dict(source=clean_bundles_file,
                parameters=get_default_args(afq_profile))

    return profile_dframe, meta


def gen_scalar_func(scalars):
    header = "def gen_scalars(scalars, "
    content = "    scalar_dict = {}\n"
    has_custom_scalar = False
    for ii, scalar in enumerate(scalars):
        if isinstance(scalar, str):
            sc = scalar.lower()
            header = header + f"{sc}_file, "
            content = content + f"    scalar_dict['{sc}'] = {sc}_file\n"
        else:
            if not has_custom_scalar:
                has_custom_scalar = True
                header = header + (
                    "subses_dict, dwi_affine, "
                    "reg_template, mapping, ")
            content = content + (
                f"    scalar_dict['{scalar.name}'] = "
                f"scalars[{ii}].get_for_subses(subses_dict, dwi_affine,"
                f" reg_template, mapping)\n")
    header = header[:-2]
    header = header + "):\n"
    content = content + "    return {'scalar_dict': scalar_dict}"

    scope = {}
    exec(header + content, scope)
    scope["gen_scalars"].__module__ = "profile"
    return pimms.calc("scalar_dict")(scope["gen_scalars"])


profile_tasks = [tract_profiles]
