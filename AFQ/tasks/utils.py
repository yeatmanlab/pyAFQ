import os.path as op

__all__ = ["get_fname", "with_name"]


def get_fname(subses_dict, suffix,
              tracking_params=None, segmentation_params=None):
    split_fdwi = op.split(subses_dict["dwi_file"])
    fname = op.join(subses_dict["results_dir"], split_fdwi[1].split('.')[0])

    if tracking_params is not None and 'odf_model' in tracking_params:
        odf_model = tracking_params['odf_model']
        directions = tracking_params['directions']
        fname = fname + (
            f'_space-RASMM_model-{odf_model}'
            f'_desc-{directions}'
        )
    if segmentation_params is not None and 'seg_algo' in segmentation_params:
        seg_algo = segmentation_params['seg_algo']
        fname = fname + f'-{seg_algo}'

    return fname + suffix


# Turn list of tasks into dictionary with names for each task
def with_name(task_list):
    task_dict = {}
    for task in task_list:
        task_dict[task.function.__name__ + "_res"] = task
    return task_dict
