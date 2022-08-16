import os.path as op
import inspect

__all__ = ["get_fname", "with_name"]


def get_fname(base_fname, suffix,
              tracking_params=None, segmentation_params=None):
    fname = base_fname
    if tracking_params is not None and 'odf_model' in tracking_params:
        odf_model = tracking_params['odf_model']
        directions = tracking_params['directions']
        fname = fname + (
            f'_space-RASMM_model-{directions+odf_model}'
        )
    if segmentation_params is not None and 'seg_algo' in segmentation_params:
        seg_algo = segmentation_params['seg_algo']
        fname = fname + f"_algo-{seg_algo}"

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
    return string.replace("-", "").replace("_", "")
