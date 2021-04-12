import nibabel as nib
import numpy as np

from pydra import mark

from AFQ.tasks.utils import *
from AFQ.definitions.utils import Definition
import AFQ.tractography as aft


@mark.task
@mark.annotate(
    {"return": {"seed_file": str}}
)
@as_file('_seed_mask.nii.gz')
def export_seed_mask(subses_tuple, tracking_params, dwi_affine):
    if isinstance(tracking_params['seed_mask'], Definition):
        seed_mask, _, seed_mask_desc =\
            tracking_params['seed_mask'].get_for_subses(subses_tuple)
    else:
        seed_mask = tracking_params['seed_mask']
        seed_mask_desc = dict(source=tracking_params['seed_mask'])
    seed_mask = nib.Nifti1Image(
        seed_mask.astype(np.float32),
        dwi_affine)
    return seed_mask, seed_mask_desc


@mark.task
@mark.annotate(
    {"return": {"stop_file": str}}
)
@as_file('_stop_mask.nii.gz')
def export_stop_mask(subses_tuple, tracking_params, dwi_affine):
    if isinstance(tracking_params['stop_mask'], Definition):
        stop_mask, _, stop_mask_desc =\
            tracking_params['stop_mask'].get_for_subses(subses_tuple)
    else:
        stop_mask = tracking_params['stop_mask']
        stop_mask_desc = dict(source=tracking_params['stop_mask'])
    stop_mask = nib.Nifti1Image(
        stop_mask.astype(np.float32),
        dwi_affine)
    return stop_mask, stop_mask_desc


@mark.task
@mark.annotate(
    {"return": {"streamlines_file": str}}
)
@as_file('_tractography.trk')
def _streamlines(subses_tuple, params_file, tracking_params={}):
    this_tracking_params = tracking_params.copy()
    if isinstance(tracking_params['seed_mask'], Definition):
        this_tracking_params['seed_mask'], _, seed_mask_desc =\
            tracking_params['seed_mask'].get_for_subses(subses_tuple)
    else:
        seed_mask_desc = dict(source=tracking_params['seed_mask'])

    if isinstance(tracking_params['stop_mask'], Definition):
        this_tracking_params['stop_mask'], _, stop_mask_desc =\
            tracking_params['stop_mask'].get_for_subses(subses_tuple)
    else:
        stop_mask_desc = dict(source=tracking_params['stop_mask'])

    start_time = time()
    sft = aft.track(params_file, **this_tracking_params)
    sft.to_vox()
    meta_directions = {"det": "deterministic",
                        "prob": "probabilistic"}
    meta = dict(
        TractographyClass="local",
        TractographyMethod=meta_directions[
            tracking_params["directions"]],
        Count=len(sft.streamlines),
        Seeding=dict(
            ROI=seed_mask_desc,
            n_seeds=tracking_params["n_seeds"],
            random_seeds=tracking_params["random_seeds"]),
        Constraints=dict(ROI=stop_mask_desc),
        Parameters=dict(
            Units="mm",
            StepSize=tracking_params["step_size"],
            MinimumLength=tracking_params["min_length"],
            MaximumLength=tracking_params["max_length"],
            Unidirectional=False),
        Timing=time() - start_time)

    return sft, meta
