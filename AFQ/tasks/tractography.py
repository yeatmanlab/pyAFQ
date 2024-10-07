import nibabel as nib
import numpy as np
from time import time
import logging

import dipy.data as dpd

import pimms
import multiprocessing

from AFQ.tasks.decorators import as_file, as_img
from AFQ.tasks.utils import with_name
from AFQ.definitions.utils import Definition
import AFQ.tractography.tractography as aft
from AFQ.tasks.utils import get_default_args
from AFQ.definitions.image import ScalarImage
from AFQ.tractography.utils import gen_seeds, get_percentile_threshold

from trx.trx_file_memmap import TrxFile
from trx.trx_file_memmap import concatenate as trx_concatenate

try:
    import ray
    has_ray = True
except ModuleNotFoundError:
    has_ray = False

try:
    from AFQ.tractography.gputractography import gpu_track
    has_gputrack = True
except ModuleNotFoundError:
    has_gputrack = False

logger = logging.getLogger('AFQ')


def _meta_from_tracking_params(
        tracking_params, start_time,
        n_streamlines, seed, stop):
    meta_directions = {
        "det": "deterministic",
        "prob": "probabilistic"}
    meta = dict(
        TractographyClass="local",
        TractographyMethod=meta_directions.get(
            tracking_params["directions"],
            tracking_params["directions"]),
        Count=n_streamlines,
        Seeding=dict(
            ROI=seed,
            n_seeds=tracking_params["n_seeds"],
            random_seeds=tracking_params["random_seeds"]),
        Constraints=dict(ROI=stop),
        Parameters=dict(
            Units="mm",
            StepSize=tracking_params["step_size"],
            MinimumLength=tracking_params["minlen"],
            MaximumLength=tracking_params["maxlen"],
            Unidirectional=False),
        Timing=time() - start_time)
    return meta


@pimms.calc("seed")
@as_file('_desc-seed_mask.nii.gz', include_track=True)
def export_seed_mask(data_imap, tracking_params):
    """
    full path to a nifti file containing the
    tractography seed mask
    """
    seed_mask = tracking_params['seed_mask']
    seed_threshold = tracking_params['seed_threshold']
    if tracking_params['thresholds_as_percentages']:
        seed_threshold = get_percentile_threshold(
            seed_mask, seed_threshold)
    seed_mask_desc = dict(
        source=tracking_params['seed_mask'],
        threshold=seed_threshold)
    return nib.Nifti1Image(seed_mask, data_imap["dwi_affine"]), \
        seed_mask_desc


@pimms.calc("stop")
@as_file('_desc-stop_mask.nii.gz', include_track=True)
def export_stop_mask(data_imap, tracking_params):
    """
    full path to a nifti file containing the
    tractography stop mask
    """
    stop_mask = tracking_params['stop_mask']
    stop_threshold = tracking_params['stop_threshold']
    if tracking_params['thresholds_as_percentages']:
        stop_threshold = get_percentile_threshold(
            stop_mask, stop_threshold)
    stop_mask_desc = dict(
        source=tracking_params['stop_mask'],
        stop_threshold=stop_threshold)
    return nib.Nifti1Image(stop_mask, data_imap["dwi_affine"]), \
        stop_mask_desc


@pimms.calc("stop")
def export_stop_mask_pft(pve_wm, pve_gm, pve_csf):
    """
    full path to a nifti file containing the
    tractography stop mask
    """
    return {"stop": [pve_wm, pve_gm, pve_csf]}


@pimms.calc("streamlines")
@as_file('_tractography', include_track=True)
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
    elif odf_model == "GQ":
        params_file = data_imap["gq_params"]
    elif odf_model == "RUMBA":
        params_file = data_imap["rumba_params"]
    else:
        raise TypeError((
            f"The ODF model you gave ({odf_model}) was not recognized"))

    # get masks
    this_tracking_params['seed_mask'] = nib.load(seed).get_fdata()
    if isinstance(stop, str):
        this_tracking_params['stop_mask'] = nib.load(stop).get_fdata()
    else:
        this_tracking_params['stop_mask'] = stop

    is_trx = this_tracking_params.get("trx", False)

    num_chunks = this_tracking_params.pop("num_chunks", False)

    if num_chunks is True:
        num_chunks = multiprocessing.cpu_count() - 1

    if is_trx:
        start_time = time()
        dtype_dict = {'positions': np.float16, 'offsets': np.uint32}
        if num_chunks and num_chunks > 1:
            if not has_ray:
                raise ImportError("Ray is required to perform tractography in"
                                  "parallel, please install ray or remove the"
                                  " 'num_chunks' arg")

            @ray.remote
            class TractActor():
                def __init__(self):
                    self.TrxFile = TrxFile
                    self.aft = aft
                    self.objects = {}

                def trx_from_lazy_tractogram(self, lazyt_id, seed, dtype_dict):
                    id = self.objects[lazyt_id]
                    return self.TrxFile.from_lazy_tractogram(
                        id,
                        seed,
                        dtype_dict=dtype_dict)

                def create_lazyt(self, id, *args, **kwargs):
                    self.objects[id] = self.aft.track(*args, **kwargs)
                    return id

                def delete_lazyt(self, id):
                    if id in self.objects:
                        del self.objects[id]
            actors = [TractActor.remote() for _ in range(num_chunks)]
            object_id = 1
            tracking_params_list = []

            # random seeds case
            if isinstance(this_tracking_params.get("n_seeds"), int) and \
               this_tracking_params.get("random_seeds"):

                remainder = this_tracking_params['n_seeds'] % num_chunks
                for i in range(num_chunks):
                    # create copy of tracking params
                    copy = this_tracking_params.copy()
                    n_seeds = this_tracking_params['n_seeds']
                    copy['n_seeds'] = n_seeds // num_chunks
                    # add remainder to 1st list
                    if i == 0:
                        copy['n_seeds'] += remainder
                    tracking_params_list.append(copy)

            elif isinstance(this_tracking_params['n_seeds'], (np.ndarray,
                                                              list)):
                n_seeds = np.array(this_tracking_params['n_seeds'])
                seed_chunks = np.array_split(n_seeds, num_chunks)
                tracking_params_list = [this_tracking_params.copy() for _ in
                                        range(num_chunks)]

                for i in range(num_chunks):
                    tracking_params_list[i]['n_seeds'] = seed_chunks[i]

            else:
                seeds = gen_seeds(
                    this_tracking_params['seed_mask'],
                    this_tracking_params['seed_threshold'],
                    this_tracking_params['n_seeds'],
                    this_tracking_params['thresholds_as_percentages'],
                    this_tracking_params['random_seeds'],
                    this_tracking_params['rng_seed'],
                    data_imap["dwi_affine"])
                seed_chunks = np.array_split(seeds, num_chunks)
                tracking_params_list = [this_tracking_params.copy() for _
                                        in range(num_chunks)]
                for i in range(num_chunks):
                    tracking_params_list[i]['n_seeds'] = seed_chunks[i]

            # create lazyt inside each actor
            tasks = [ray_actor.create_lazyt.remote(object_id, params_file,
                     **tracking_params_list[i]) for i, ray_actor in
                     enumerate(actors)]
            ray.get(tasks)

            # create trx from lazyt
            tasks = [ray_actor.trx_from_lazy_tractogram.remote(object_id, seed,
                     dtype_dict=dtype_dict) for ray_actor in actors]
            sfts = ray.get(tasks)

            # cleanup objects
            tasks = [ray_actor.delete_lazyt.remote(object_id) for ray_actor in
                     actors]
            ray.get(tasks)

            sft = trx_concatenate(sfts)
        else:
            lazyt = aft.track(params_file, **this_tracking_params)
            sft = TrxFile.from_lazy_tractogram(
                lazyt,
                seed,
                dtype_dict=dtype_dict)
        n_streamlines = len(sft)

    else:
        start_time = time()
        sft = aft.track(params_file, **this_tracking_params)
        sft.to_vox()
        n_streamlines = len(sft.streamlines)

    return sft, _meta_from_tracking_params(
        tracking_params, start_time,
        n_streamlines, seed, stop)


@pimms.calc("streamlines")
def custom_tractography(import_tract=None):
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
    if not isinstance(import_tract, str):
        raise TypeError(
            "import_tract must be"
            + " either a dict or a str")
    return import_tract


@pimms.calc("streamlines")
@as_file('_tractography', include_track=True)
def gpu_tractography(data_imap, tracking_params, seed, stop,
                     tractography_ngpus=0, chunk_size=100000):
    """
    full path to the complete, unsegmented tractography file

    Parameters
    ----------
    tractography_ngpus : int, optional
        Number of GPUs to use in tractography. If non-0,
        this algorithm is used for tractography,
        https://github.com/dipy/GPUStreamlines
        PTT, Prob can be used with any SHM model.
        Bootstrapped can be done with CSA/OPDT.
        Default: 0
    chunk_size : int, optional
        Chunk size for GPU tracking.
        Default: 100000
    """
    start_time = time()
    if tracking_params["directions"] == "boot":
        data = data_imap["data"]
    else:
        data = nib.load(
            data_imap[tracking_params["odf_model"].lower() + "_params"]).get_fdata()

    sphere = tracking_params["sphere"]
    if sphere is None:
        sphere = dpd.default_sphere

    sft = gpu_track(
        data, data_imap["gtab"],
        nib.load(seed), nib.load(stop),
        tracking_params["odf_model"],
        sphere,
        tracking_params["directions"],
        tracking_params["seed_threshold"],
        tracking_params["stop_threshold"],
        tracking_params["thresholds_as_percentages"],
        tracking_params["max_angle"], tracking_params["step_size"],
        tracking_params["n_seeds"],
        tracking_params["random_seeds"],
        tracking_params["rng_seed"],
        tracking_params["trx"],
        tractography_ngpus,
        chunk_size)

    return sft, _meta_from_tracking_params(
        tracking_params, start_time,
        sft, seed, stop)


def get_tractography_plan(kwargs):
    if "tracking_params" in kwargs\
            and not isinstance(kwargs["tracking_params"], dict):
        raise TypeError(
            "tracking_params a dict")

    tractography_tasks = with_name([
        export_seed_mask, export_stop_mask, streamlines])

    # use GPU accelerated tractography if asked for
    if "tractography_ngpus" in kwargs and kwargs["tractography_ngpus"] != 0:
        if not has_gputrack:
            raise ImportError("Please install from ghcr.io/nrdg/pyafq_gpu"
                              " docker file to use gpu-accelerated"
                              "tractography")
        tractography_tasks["streamlines_res"] = gpu_tractography
    # use imported tractography if given
    if "import_tract" in kwargs and kwargs["import_tract"] is not None:
        tractography_tasks["streamlines_res"] = custom_tractography

    # determine reasonable defaults
    best_scalar = kwargs["scalars"][0]
    fa_found = False
    for scalar in kwargs["scalars"]:
        if isinstance(scalar, str):
            if "fa" in scalar:
                best_scalar = scalar
                fa_found = True
                break
        else:
            if "fa" in scalar.get_name():
                best_scalar = scalar
                fa_found = True
                break
    if not fa_found:
        logger.warning(
            "FA not found in list of scalars, will use first scalar"
            " for the seed and stop mask and visualizations"
            " unless these are also specified")
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
        logger.info((
            "No seed mask given, using FA (or first scalar if none are FA)"
            "thresholded to 0.2"))
    if kwargs["tracking_params"]["stop_mask"] is None:
        kwargs["tracking_params"]["stop_mask"] = ScalarImage(
            kwargs["best_scalar"])
        kwargs["tracking_params"]["stop_threshold"] = 0.2
        logger.info((
            "No stop mask given, using FA (or first scalar if none are FA)"
            "thresholded to 0.2"))

    stop_mask = kwargs["tracking_params"]['stop_mask']
    seed_mask = kwargs["tracking_params"]['seed_mask']

    if kwargs["tracking_params"]["tracker"] == "pft":
        probseg_funcs = stop_mask.get_image_getter("tractography")
        tractography_tasks["wm_res"] = pimms.calc("pve_wm")(probseg_funcs[0])
        tractography_tasks["gm_res"] = pimms.calc("pve_gm")(probseg_funcs[1])
        tractography_tasks["csf_res"] = pimms.calc("pve_csf")(probseg_funcs[2])
        tractography_tasks["export_stop_mask_res"] = \
            export_stop_mask_pft
    elif isinstance(stop_mask, Definition):
        tractography_tasks["export_stop_mask_res"] = pimms.calc("stop")(
            as_file('_desc-stop_mask.nii.gz', include_track=True)(
                stop_mask.get_image_getter("tractography")))

    if isinstance(seed_mask, Definition):
        tractography_tasks["export_seed_mask_res"] = pimms.calc("seed")(
            as_file('_desc-seed_mask.nii.gz', include_track=True)(
                seed_mask.get_image_getter("tractography")))

    return pimms.plan(**tractography_tasks)
