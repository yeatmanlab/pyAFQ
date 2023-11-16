import os.path as op
import os
import json


def write_json(fname, data):
    """
    Write data to JSON file.

    Parameters
    ----------
    fname : str
        Full path to the file to write.

    data : dict
        A dict containing the data to write.

    Returns
    -------
    None
    """
    with open(fname, 'w') as ff:
        json.dump(data, ff, default=lambda obj: "Not Serializable")


def read_json(fname):
    """
    Read data from a JSON file.

    Parameters
    ----------
    fname : str
        Full path to the data-containing file

    Returns
    -------
    dict
    """
    with open(fname, 'r') as ff:
        out = json.load(ff)
    return out


def drop_extension(path):
    base_fname = op.basename(path).split('.')[0]
    return path.split(base_fname)[0] + base_fname


def apply_cmd_to_afq_derivs(
        derivs_dir, base_fname, cmd="rm", exception_file_names=[], suffix="",
        dependent_on=None):
    if dependent_on is None:
        dependent_on_list = ["dwi", "trk", "rec", "prof"]
    elif dependent_on.lower() == "track":
        dependent_on_list = ["trk", "rec", "prof"]
    elif dependent_on.lower() == "recog":
        dependent_on_list = ["rec", "prof"]
    elif dependent_on.lower() == "prof":
        dependent_on_list = ["prof"]
    else:
        raise ValueError((
            "dependent_on must be one of "
            "None, 'track', 'recog', 'prof'."))

    for filename in os.listdir(derivs_dir):
        full_path = os.path.join(derivs_dir, filename)
        if os.path.isfile(full_path) or os.path.islink(full_path):
            if (full_path in exception_file_names)\
                    or (not full_path.startswith(base_fname))\
                    or filename.endswith("json"):
                continue
            sidecar_file = f'{drop_extension(full_path)}.json'
            if op.exists(sidecar_file):
                sidecar_info = read_json(sidecar_file)
                if "dependent" in sidecar_info\
                    and sidecar_info["dependent"]\
                        in dependent_on_list:
                    os.system(f"{cmd} {full_path} {suffix}")
                    os.system(f"{cmd} {sidecar_file} {suffix}")
            else:
                os.system(f"{cmd} {full_path} {suffix}")
        elif os.path.isdir(full_path):
            # other than ROIs, folders are dependent on everything
            if dependent_on is None or filename != "ROIs":
                os.system(f"{cmd} -r {full_path} {suffix}")
