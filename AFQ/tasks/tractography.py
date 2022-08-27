import nibabel as nib
from time import time
import logging

import pimms

from AFQ.tasks.decorators import as_file, as_img
from AFQ.tasks.utils import with_name
from AFQ.definitions.utils import Definition
import AFQ.tractography as aft
from AFQ.tasks.utils import get_default_args
from AFQ.definitions.image import ScalarImage

logger = logging.getLogger('AFQ')


@pimms.calc("seed")
@as_file('_desc-seed_mask.nii.gz', include_track=True)
@as_img
def export_seed_mask(tracking_params):
    """
    full path to a nifti file containing the
    tractography seed mask
    """
    seed_mask = tracking_params['seed_mask']
    seed_mask_desc = dict(source=tracking_params['seed_mask'])
    return seed_mask, seed_mask_desc


@pimms.calc("stop")
@as_file('_desc-stop_mask.nii.gz', include_track=True)
@as_img
def export_stop_mask(tracking_params):
    """
    full path to a nifti file containing the
    tractography stop mask
    """
    stop_mask = tracking_params['stop_mask']
    stop_mask_desc = dict(source=tracking_params['stop_mask'])
    return stop_mask, stop_mask_desc


@pimms.calc("stop")
def export_stop_mask_pft(pve_wm, pve_gm, pve_csf):
    """
    full path to a nifti file containing the
    tractography stop mask
    """
    return {"stop": [pve_wm, pve_gm, pve_csf]}


@pimms.calc("streamlines")
@as_file('_tractography.trk', include_track=True)
def streamlines(data_imap, seed, stop,
                tracking_params):
    """
    full path to the complete, unsegmented tractography file

    Parameters
    ----------
    tracking_params : dict, optional
        The parameters for tracking. Default: use the default behavior of
        the aft.track function. Seed mask and seed threshold, if not
        specified, are replaced with scalar masks from scalar[0]
        thresholded to 0.2. The ``seed_mask`` and ``stop_mask`` items of
        this dict may be ``AFQ.definitions.image.ImageFile`` instances.
        If ``tracker`` is set to "pft" then ``stop_mask`` should be
        an instance of ``AFQ.definitions.image.PFTImage``.
    """
    this_tracking_params = tracking_params.copy()

    # get odf_model
    odf_model = this_tracking_params["odf_model"]
    if odf_model == "DTI":
        params_file = data_imap["dti_params"]
    elif odf_model == "CSD":
        params_file = data_imap["csd_params"]
    elif odf_model == "DKI":
        params_file = data_imap["dki_params"]
    else:
        raise TypeError((
            f"The ODF model you gave ({odf_model}) was not recognized"))

    # get masks
    this_tracking_params['seed_mask'] = nib.load(seed).get_fdata()
    if isinstance(stop, str):
        this_tracking_params['stop_mask'] = nib.load(stop).get_fdata()
    else:
        this_tracking_params['stop_mask'] = stop

    # perform tractography
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
            ROI=seed,
            n_seeds=tracking_params["n_seeds"],
            random_seeds=tracking_params["random_seeds"]),
        Constraints=dict(ROI=stop),
        Parameters=dict(
            Units="mm",
            StepSize=tracking_params["step_size"],
            MinimumLength=tracking_params["min_length"],
            MaximumLength=tracking_params["max_length"],
            Unidirectional=False),
        Timing=time() - start_time)

    return sft, meta


@pimms.calc("streamlines")
def custom_tractography(bids_info, import_tract=None):
    """
    full path to the complete, unsegmented tractography file

    Parameters
    ----------
    import_tract : dict or str or None, optional
        BIDS filters for inputing a user made tractography file,
        or a path to the tractography file. If None, DIPY is used
        to generate the tractography.
        Default: None
    """
    if not isinstance(import_tract, dict) and\
            not isinstance(import_tract, str):
        raise TypeError(
            "import_tract must be"
            + " either a dict or a str")
    if isinstance(import_tract, dict):
        if bids_info is None:
            raise ValueError((
                "bids_info must be provided if using"
                " bids filters to find imported tracts"))
        import_tract = \
            bids_info["bids_layout"].get(
                subject=bids_info["subject"],
                session=bids_info["session"],
                extension=[
                    '.trk',
                    '.tck',
                    '.vtk',
                    '.fib',
                    '.dpy'],
                return_type='filename',
                **import_tract)
        if len(import_tract) < 1:
            raise ValueError((
                "No custom tractography found for subject "
                f"{bids_info['subject']} and session "
                f"{bids_info['session']}."))
        elif len(import_tract) > 1:
            import_tract = import_tract[0]
            logger.warning((
                f"Multiple viable custom tractographies found for"
                f" subject "
                f"{bids_info['subject']} and session "
                f"{bids_info['session']}. Will use: {import_tract}"))
        else:
            import_tract = import_tract[0]
    return import_tract


def get_tractography_plan(kwargs):
    if "tracking_params" in kwargs\
            and not isinstance(kwargs["tracking_params"], dict):
        raise TypeError(
            "tracking_params a dict")

    tractography_tasks = with_name([
        export_seed_mask, export_stop_mask, streamlines])

    # use imported tractography if given
    if "import_tract" in kwargs and kwargs["import_tract"] is not None:
        tractography_tasks["streamlines_res"] = custom_tractography

    # determine reasonable defaults
    best_scalar = kwargs["scalars"][0]
    for scalar in kwargs["scalars"]:
        if isinstance(scalar, str):
            if "fa" in scalar:
                best_scalar = scalar
                break
        else:
            if "fa" in scalar.get_name():
                best_scalar = scalar
                break
    kwargs["best_scalar"] = best_scalar

    default_tracking_params = get_default_args(aft.track)

    # Replace the defaults only for kwargs for which a non-default value
    # was given:
    if "tracking_params" in kwargs:
        for k in kwargs["tracking_params"]:
            default_tracking_params[k] = kwargs["tracking_params"][k]

    kwargs["tracking_params"] = default_tracking_params
    kwargs["tracking_params"]["odf_model"] =\
        kwargs["tracking_params"]["odf_model"].upper()
    if kwargs["tracking_params"]["seed_mask"] is None:
        kwargs["tracking_params"]["seed_mask"] = ScalarImage(
            kwargs["best_scalar"])
        kwargs["tracking_params"]["seed_threshold"] = 0.2
    if kwargs["tracking_params"]["stop_mask"] is None:
        kwargs["tracking_params"]["stop_mask"] = ScalarImage(
            kwargs["best_scalar"])
        kwargs["tracking_params"]["stop_threshold"] = 0.2

    stop_mask = kwargs["tracking_params"]['stop_mask']
    seed_mask = kwargs["tracking_params"]['seed_mask']
    dwi = kwargs["dwi"]
    bids_info = kwargs["bids_info"]

    if bids_info is not None:
        if isinstance(stop_mask, Definition):
            stop_mask.find_path(
                bids_info["bids_layout"],
                dwi,
                bids_info["subject"],
                bids_info["session"])
        if isinstance(seed_mask, Definition):
            seed_mask.find_path(
                bids_info["bids_layout"],
                dwi,
                bids_info["subject"],
                bids_info["session"])

    if kwargs["tracking_params"]["tracker"] == "pft":
        probseg_funcs = stop_mask.get_image_getter("tractography")
        tractography_tasks["wm_res"] = pimms.calc("pve_wm")(probseg_funcs[0])
        tractography_tasks["gm_res"] = pimms.calc("pve_gm")(probseg_funcs[1])
        tractography_tasks["csf_res"] = pimms.calc("pve_csf")(probseg_funcs[2])
        tractography_tasks["export_stop_mask_res"] = \
            export_stop_mask_pft
    elif isinstance(stop_mask, Definition):
        tractography_tasks["export_stop_mask_res"] =\
            pimms.calc("stop")(as_file('_desc-stop_mask.nii.gz')(
                stop_mask.get_image_getter("tractography")))

    if isinstance(seed_mask, Definition):
        tractography_tasks["export_seed_mask_res"] = pimms.calc("seed")(
            as_file('_desc-seed_mask.nii.gz')(
                seed_mask.get_image_getter("tractography")))

    return pimms.plan(**tractography_tasks)
