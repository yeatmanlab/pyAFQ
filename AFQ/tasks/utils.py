from AFQ.utils.path import drop_extension
import os.path as op
import inspect

__all__ = ["get_fname", "with_name", "get_base_fname"]


def get_base_fname(output_dir, dwi_data_file):
    # setup up path and base file name for outputs
    # remove suffix and desc from dwi data file name
    used_key_list = ["desc", "space", "to", "from"]
    dwi_dfile_no_ext = op.join(
        output_dir,
        drop_extension(op.basename(dwi_data_file)))
    fname = op.dirname(dwi_dfile_no_ext) + "/"
    for key_val_pair in op.basename(dwi_dfile_no_ext).split("_"):
        if "-" in key_val_pair:
            key = key_val_pair.split("-")[0]
            if key not in used_key_list:
                fname = fname + key_val_pair + "_"
    fname = fname[:-1]
    return fname


def get_fname(base_fname, suffix,
              tracking_params=None, segmentation_params=None):
    fname = base_fname
    if tracking_params is not None and 'odf_model' in tracking_params:
        odf_model = tracking_params['odf_model']
        directions = tracking_params['directions']
        fname = fname + (
            f'_coordsys-RASMM_trkmethod-{directions+odf_model}'
        )
    if segmentation_params is not None and 'seg_algo' in segmentation_params:
        seg_algo = segmentation_params['seg_algo']
        fname = fname + f"_recogmethod-{seg_algo}"

    return fname + suffix


# Turn list of tasks into dictionary with names for each task
def with_name(task_list):
    return {f"{task.function.__name__}_res": task for task in task_list}


def get_default_args(func):
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def str_to_desc(string):
    return string.replace("-", "").replace("_", "").replace(" ", "")
