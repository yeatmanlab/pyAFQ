import toml
import datetime
import platform
import os.path as op

from argparse import ArgumentParser
from funcargparse import FuncArgParser

from AFQ.definitions.image import *  # interprets masks loaded from toml
from AFQ.definitions.mapping import *  # interprets mappings loaded from toml
from AFQ.api.bundle_dict import *  # interprets bundle_dicts loaded from toml
from AFQ.definitions.utils import Definition
from AFQ.api.utils import kwargs_descriptors

import nibabel as nib  # allows users to input nibabel objects


def model_input_parser(usage):
    parser = ArgumentParser(usage)

    parser.add_argument("-d", "--dwi", dest='dwi', action="append",
                        help="DWI files (enter one or more)")

    parser.add_argument("-l", "--bval", dest="bval", action="append",
                        help="B-value files (enter one or more)")

    parser.add_argument("-c", "--bvec", dest="bvec", action="append",
                        help="B-vector files (enter one or more)")

    parser.add_argument("-o", "--out_dir", dest="out_dir", action="store",
                        help="""Full path to directory for files to be saved
                            (will be created if it doesn't exist)")""")

    parser.add_argument("-m", "--mask", dest="mask", action="store",
                        default=None, help="Mask file")

    parser.add_argument('-b', '--b0_threshold', dest="b0_threshold",
                        action="store", help="b0 threshold", default=0)

    return parser


def model_predict_input_parser(usage):
    parser = ArgumentParser(usage)

    parser.add_argument("-p", "--params", dest="params", action="store",
                        help="A file containing model params")

    parser.add_argument("-l", "--bval", dest="bval", action="append",
                        help="B-value files (enter one or more)")

    parser.add_argument("-c", "--bvec", dest="bvec", action="append",
                        help="B-vector files (enter one or more)")

    parser.add_argument("-o", "--out_dir", dest="out_dir", action="store",
                        help="""Full path to directory for files to be saved
                            (will be created if it doesn't exist)")""")

    parser.add_argument(
        "-s", "--S0_file", dest="S0_file", action="store",
        help="File containing S0 measurements to use in prediction")

    parser.add_argument('-b', '--b0_threshold', dest="b0_threshold",
                        help="b0 threshold (default: 0)",
                        action="store", default=0)
    return parser


def toml_to_val(t):
    if isinstance(t, str) and len(t) < 1:
        return None
    elif isinstance(t, list):
        ls = []
        for e in t:
            ls.append(toml_to_val(e))
        return ls
    elif isinstance(t, str) and t[0] == '[':
        return eval(t)
    elif isinstance(t, str) and t[0] == '{':
        return eval(t)  # interpret as dictionary
    elif isinstance(t, str) and ("Image" in t or "Map" in t or "Dict" in t):
        try:
            definition_or_dict = eval(t)
        except NameError:
            return t
        if isinstance(definition_or_dict, Definition):
            return definition_or_dict
        elif isinstance(definition_or_dict, BundleDict):
            return definition_or_dict
        else:
            return t
    else:
        return t


def val_to_toml(v):
    if v is None:
        return '""'
    elif isinstance(v, Definition):
        return f'"{v.str_for_toml()}"'
    elif isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, bool):
        if v:
            return "true"
        else:
            return "false"
    elif callable(v):
        return f'"{v.__name__}"'
    elif isinstance(v, dict):
        return f'"{v}"'
    elif isinstance(v, list):
        return f'"{v}"'
    else:
        return f"{v}"


def dict_to_toml(dictionary):
    toml = '# Use \'\' to indicate None\n# Wrap dictionaries in quotes\n'
    toml = toml + '# Wrap definition object instantiations in quotes\n\n'
    for section, args in dictionary.items():
        if section == "AFQ_desc":
            toml = "# " + dictionary["AFQ_desc"].replace("\n", "\n# ")\
                + "\n\n" + toml
            continue
        toml = toml + f'[{section}]\n'
        for arg, arg_info in args.items():
            toml = toml + '\n'
            if isinstance(arg_info, dict):
                if 'desc' in arg_info:
                    toml = toml + arg_info['desc']
                toml = toml + f"{arg} = {val_to_toml(arg_info['default'])}\n"
            else:
                toml = toml + f"{arg} = {val_to_toml(arg_info)}\n"
        toml = toml + '\n'
    return toml + '\n'


# these params are handled internally in the qsiprep pipeline,
# not shown to the user (mostly BIDS filters stuff)
qsi_prep_ignore_params = [
    "bids_path", "bids_filters", "preproc_pipeline",
    "participant_labels", "output_dir"]


def dict_to_json(dictionary):
    json = "                "
    local_ignore = qsi_prep_ignore_params.copy()
    for section, args in dictionary.items():
        if section == "AFQ_desc":
            continue
        for arg, arg_info in args.items():
            if arg in local_ignore:
                continue
            local_ignore.append(arg)
            if isinstance(arg_info, dict):
                json = json\
                    + f'"{arg}": {val_to_toml(arg_info["default"])}'
            else:
                json = json + f'"{arg}": {val_to_toml(arg_info)}'
            json = json + ',\n                '
    return json[:-18]  # remove trailing ,\n and indent


def func_dict_to_arg_dict(func_dict=None, logger=None):
    if func_dict is None:
        import AFQ.segmentation as seg
        import AFQ.tractography as aft
        from AFQ.api.group import GroupAFQ

        func_dict = {
            "BIDS": GroupAFQ.__init__,
            "Tractography": aft.track,
            "Segmentation": seg.Segmentation.__init__,
            "Cleaning": seg.clean_bundle}

    arg_dict = {}
    for name, func in func_dict.items():
        docstr_parser = FuncArgParser()
        docstr_parser.setup_args(func)
        if name == "BIDS":
            arg_dict["AFQ_desc"] = docstr_parser.description
        for arg, info in docstr_parser.unfinished_arguments.items():
            try:
                section = name.upper() + "_PARAMS"
                desc = info['help']
                if name != "BIDS" and 'positional' in info and info[
                        'positional']:
                    continue
            except (KeyError, IndexError) as error:
                if logger is not None:
                    logger.error(
                        "We are missing a valid description for the "
                        + f"{name} argument {arg}")
                raise error
            if section not in arg_dict.keys():
                arg_dict[section] = {}
            arg_dict[section][arg] = {}
            if 'default' in info:
                default = info['default']
            else:
                default = None
            arg_dict[section][arg]['default'] = default
            arg_dict[section][arg]['desc'] = desc

    for section, arg_info in kwargs_descriptors.items():
        section = section.upper()
        if section not in arg_dict.keys():
            arg_dict[section] = {}
        for arg, info in arg_info.items():
            if arg not in [
                    "clean_params", "segmentation_params",
                    "tracking_params"]:
                arg_dict[section][arg] = info

    for section, arg_info in arg_dict.items():
        if section == "AFQ_desc":
            continue
        for arg, info in arg_info.items():
            desc = arg_dict[section][arg]['desc']
            arg_dict[section][arg]['desc'] = ''
            for desc_line in desc.splitlines():
                f_desc_line = '# ' + desc_line.strip() + '\n'
                arg_dict[section][arg]['desc'] = \
                    arg_dict[section][arg]['desc'] + f_desc_line

    return arg_dict


def parse_config_run_afq(toml_file, default_arg_dict, to_call="export_all",
                         overwrite=False,
                         logger=None,
                         verbose=False,
                         dry_run=False,
                         special_args={
                             "CLEANING_PARAMS": "clean_params",
                             "SEGMENTATION_PARAMS": "segmentation_params",
                             "TRACTOGRAPHY_PARAMS": "tracking_params"}):
    from AFQ.api.group import GroupAFQ
    from AFQ import __version__
    # load configuration file
    if not op.exists(toml_file):
        raise FileExistsError(
            "Config file does not exist. "
            + "If you want to generate this file,"
            + " add the argument --generate-config-only")
    f_arg_dict = toml.load(toml_file)

    # extract arguments from file
    kwargs = {}
    bids_path = ''
    for section, args in f_arg_dict.items():
        for arg, default in args.items():
            if section not in default_arg_dict:
                default_arg_dict[section] = {}
            if arg == 'bids_path':
                bids_path = default
            else:
                val = toml_to_val(default)
                is_special = False
                for toml_key, doc_arg in special_args.items():
                    if section == toml_key:
                        if doc_arg not in kwargs:
                            kwargs[doc_arg] = {}
                        kwargs[doc_arg][arg] = val
                        is_special = True
                if not is_special:
                    kwargs[arg] = val
            if arg not in default_arg_dict[section]:
                default_arg_dict[section][arg] = {}
            default_arg_dict[section][arg]['default'] = default

    if logger is not None and (verbose or dry_run):
        logger.info("The following arguments are recognized: " + str(kwargs))

    if dry_run:
        return

    # if overwrite, write new file with updated docs / args
    if overwrite:
        if logger is not None:
            logger.info("Updating configuration file.")
        with open(toml_file, 'w') as ff:
            ff.write(dict_to_toml(default_arg_dict))

    if bids_path == '':
        raise RuntimeError("Config file must provide bids_path")

    # generate metadata file for this run
    default_arg_dict['pyAFQ'] = {}
    default_arg_dict['pyAFQ']['utc_time_started'] = \
        datetime.datetime.now().isoformat('T')
    default_arg_dict['pyAFQ']['version'] = __version__
    default_arg_dict['pyAFQ']['platform'] = platform.system()

    myafq = GroupAFQ(bids_path, **kwargs)

    afq_metadata_file = op.join(myafq.afq_path, 'afq_metadata.toml')
    with open(afq_metadata_file, 'w') as ff:
        ff.write(dict_to_toml(default_arg_dict))

    # call user specified function:
    if to_call == "all":
        myafq.export_all()
    else:
        myafq.export(to_call)

    # If you got this far, you can report on time ended and record that:
    default_arg_dict['pyAFQ']['utc_time_ended'] = datetime.datetime.now(
    ).isoformat('T')
    with open(afq_metadata_file, 'w') as ff:
        ff.write(dict_to_toml(default_arg_dict))


def generate_config(toml_file, default_arg_dict, overwrite=False,
                    logger=None):
    if not overwrite and op.exists(toml_file):
        raise FileExistsError(
            "Config file already exists. "
            + "If you want to overwrite this file,"
            + " add the argument --overwrite-config")
    if logger is not None:
        logger.info("Generating default configuration file.")
    toml_file = open(toml_file, 'w')
    toml_file.write(dict_to_toml(default_arg_dict))
    toml_file.close()


def generate_json(json_folder, overwrite=False,
                  logger=None):
    json_file_our_trk = op.join(json_folder, "pyafq.json")
    json_file_their_trk = op.join(json_folder, "pyafq_input_trk.json")
    if not overwrite and (
            op.exists(json_file_our_trk) or op.exists(json_file_their_trk)):
        raise FileExistsError(
            "Config file already exists. "
            + "If you want to overwrite this file,"
            + " add the argument --overwrite-config")
    if logger is not None:
        logger.info("Generating pyAFQ full pipeline QSIprep json file.")
    qsi_spec_intro_our_trk = """{
    "description": "Use pyAFQ to perform the full Tractometry pipeline",
    "space": "T1w",
    "name": "pyAFQ_full",
    "atlases": [],
    "nodes": [
        {
            "name": "pyAFQ_full",
            "software": "pyAFQ",
            "action": "pyAFQ_full",
            "input": "qsiprep",
            "output_suffix": "PYAFQ_FULL",
            "parameters": {
                "use_external_tracking": false,
                "export": "all",
"""
    qsi_spec_intro_their_trk = """{
    "description": "Use pyAFQ to perform the Tractometry pipeline, with tractography from qsiprep",
    "space": "T1w",
    "name": "pyAFQ_import_trk",
    "atlases": [],
    "nodes": [
        {
            "name": "msmt_csd",
            "software": "MRTrix3",
            "action": "csd",
            "output_suffix": "msmtcsd",
            "input": "qsiprep",
            "parameters": {
                "mtnormalize": true,
                "response": {
                "algorithm": "dhollander"
                },
                "fod": {
                "algorithm": "msmt_csd",
                "max_sh": [4, 8, 8]
                }
            }
        },
        {
            "name": "track_ifod2",
            "software": "MRTrix3",
            "action": "tractography",
            "output_suffix": "ifod2",
            "input": "msmt_csd",
            "parameters": {
                "use_5tt": false,
                "use_sift2": true,
                "tckgen":{
                "algorithm": "iFOD2",
                "select": 1e6,
                "max_length": 250,
                "min_length": 30,
                "power":0.33
                },
                "sift2":{}
            }
        },
        {
            "name": "pyAFQ_full",
            "software": "pyAFQ",
            "action": "pyAFQ_full",
            "input": "track_ifod2",
            "output_suffix": "PYAFQ_FULL_ET",
            "parameters": {
                "use_external_tracking": true,
                "export": "all",
"""  # noqa
    qsi_spec_outro = """
            }
        }
    ]
}"""
    import AFQ.segmentation as seg
    import AFQ.tractography as aft

    func_dict = {
        "Tractography": aft.track,
        "Segmentation": seg.Segmentation.__init__,
        "Cleaning": seg.clean_bundle}

    arg_dict = func_dict_to_arg_dict(func_dict, logger=logger)

    json_file = open(json_file_our_trk, 'w')
    json_file.write(qsi_spec_intro_our_trk)
    json_file.write(dict_to_json(arg_dict))
    json_file.write(qsi_spec_outro)
    json_file.close()

    json_file = open(json_file_their_trk, 'w')
    json_file.write(qsi_spec_intro_their_trk)
    json_file.write(dict_to_json(arg_dict))
    json_file.write(qsi_spec_outro)
    json_file.close()
