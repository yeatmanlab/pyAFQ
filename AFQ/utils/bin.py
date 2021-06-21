import inspect
import toml
import datetime
import platform
import os.path as op
import os

from argparse import ArgumentParser
from funcargparse import FuncArgParser

from AFQ.definitions.mask import *  # interprets masks loaded from toml
from AFQ.definitions.mapping import *  # interprets mappings loaded from toml
from AFQ.definitions.scalar import *  # interprets scalars loaded from toml
from AFQ.definitions.utils import Definition

import nibabel as nib  # allows users to input nibabel objects


def parse_string(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


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


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def toml_to_val(t):
    if isinstance(t, str) and len(t) < 1:
        return None
    elif isinstance(t, list):
        ls = []
        for e in t:
            ls.append(toml_to_val(e))
        return ls
    elif isinstance(t, str) and t[0] == '{':
        return eval(t)  # interpret as dictionary
    elif isinstance(t, str) and ("Mask" in t or "Map" in t or "Scalar" in t):
        try:
            definition = eval(t)
        except NameError:
            return t
        if isinstance(definition, Definition):
            return definition
        else:
            return t
    else:
        return t


def val_to_toml(v):
    if v is None:
        return "''"
    elif isinstance(v, Definition):
        return f"'{v.str_for_toml()}'"
    elif isinstance(v, str):
        return f"'{v}'"
    elif isinstance(v, bool):
        if v:
            return "true"
        else:
            return "false"
    elif callable(v):
        return f"'{v.__name__}'"
    elif isinstance(v, dict):
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


def func_dict_to_arg_dict(func_dict=None, logger=None):
    if func_dict is None:
        import AFQ.segmentation as seg
        import AFQ.tractography as aft
        import AFQ.api as api

        func_dict = {
            "AFQ": api.AFQ.__init__,
            "Tractography": aft.track,
            "Segmentation": seg.Segmentation.__init__,
            "Cleaning": seg.clean_bundle}

    arg_dict = {}
    for name, func in func_dict.items():
        docstr_parser = FuncArgParser()
        docstr_parser.setup_args(func)
        if name == "AFQ":
            arg_dict["AFQ_desc"] = docstr_parser.description
        for arg, info in docstr_parser.unfinished_arguments.items():
            try:
                if name == "AFQ":
                    if arg in [
                            "tracking_params",
                            "segmentation_params",
                            "clean_params"]:
                        continue
                    section, desc = info['help'].split('[')[1].split(']')
                else:
                    section = name.upper()
                    desc = info['help']
                    if 'positional' in info and info['positional']:
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
                         special_args={"CLEANING": "clean_params",
                                       "SEGMENTATION": "segmentation_params",
                                       "TRACTOGRAPHY": "tracking_params"}):
    import AFQ.api as api
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
            elif section == "KWARGS":
                kwargs[arg] = toml_to_val(default)
            elif arg in default_arg_dict[section]:
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

    if logger is not None and verbose:
        logger.info("The following arguments are recognized: " + str(kwargs))

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

    afq_path = op.join(bids_path, 'derivatives', 'afq')
    os.makedirs(afq_path, exist_ok=True)

    afq_metadata_file = op.join(afq_path, 'afq_metadata.toml')
    with open(afq_metadata_file, 'w') as ff:
        ff.write(dict_to_toml(default_arg_dict))

    myafq = api.AFQ(bids_path, **kwargs)

    # call user specified function:
    getattr(myafq, to_call)()

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
