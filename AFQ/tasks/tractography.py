import nibabel as nib
import numpy as np
from time import time

import pimms

from AFQ.tasks.utils import as_file, as_img
from AFQ.definitions.utils import Definition
import AFQ.tractography as aft


@pimms.calc("seed_file")
@as_file('_seed_mask.nii.gz')
@as_img
def export_seed_mask(subses_dict, dwi_affine, tracking_params):
    seed_mask = tracking_params['seed_mask']
    seed_mask_desc = dict(source=tracking_params['seed_mask'])
    return seed_mask, seed_mask_desc


@pimms.calc("stop_file")
@as_file('_stop_mask.nii.gz')
@as_img
def export_stop_mask(subses_dict, dwi_affine, tracking_params):
    stop_mask = tracking_params['stop_mask']
    stop_mask_desc = dict(source=tracking_params['stop_mask'])
    return stop_mask, stop_mask_desc


@pimms.calc("stop_file")
def export_stop_mask_pft(pve_wm, pve_gm, pve_csf):
    return {"stop_file": [pve_wm, pve_gm, pve_csf]}


@pimms.calc("streamlines_file")
@as_file('_tractography.trk', include_track=True)
def streamlines(subses_dict, params_file, seed_file, stop_file,
                tracking_params):
    this_tracking_params = tracking_params.copy()
    this_tracking_params['seed_mask'] = nib.load(seed_file).get_fdata()
    if isinstance(stop_file, str):
        this_tracking_params['stop_mask'] = nib.load(stop_file).get_fdata()
    else:
        this_tracking_params['stop_mask'] = stop_file

    start_time = time()
    sft = aft.track(params_file, **this_tracking_params)
    sft.to_vox()
    meta_directions = {
        "det": "deterministic",
        "prob": "probabilistic"}
    meta = dict(
        TractographyClass="local",
        TractographyMethod=meta_directions[
            tracking_params["directions"]],
        Count=len(sft.streamlines),
        Seeding=dict(
            ROI=seed_file,
            n_seeds=tracking_params["n_seeds"],
            random_seeds=tracking_params["random_seeds"]),
        Constraints=dict(ROI=stop_file),
        Parameters=dict(
            Units="mm",
            StepSize=tracking_params["step_size"],
            MinimumLength=tracking_params["min_length"],
            MaximumLength=tracking_params["max_length"],
            Unidirectional=False),
        Timing=time() - start_time)

    return sft, meta


@pimms.calc("streamlines_file")
def custom_tractography(custom_tract_file):
    return custom_tract_file


tractography_tasks = [
    export_seed_mask, export_stop_mask, streamlines]
