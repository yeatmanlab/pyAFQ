import inspect
import toml
from argparse import ArgumentParser
from funcargparse import FuncArgParser
from ast import literal_eval


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
    elif isinstance(t, str) and t[0] == '{':
        return literal_eval(t)  # interpret as dictionary
    else:
        return t


def val_to_toml(v):
    if v is None:
        return "''"
    if isinstance(v, str):
        return f"'{v}'"
    elif isinstance(v, bool):
        if v:
            return "true"
        else:
            return "false"
    elif callable(v):
        return f"'{v.__name__}'"
    elif isinstance(v, dict):
        return f"\"{v}\""
    else:
        return f"{v}"


def dict_to_toml(dictionary):
    toml = ''
    for section, args in dictionary.items():
        toml = toml + f'[{section}]\n'
        for arg, arg_info in args.items():
            toml = toml + '\n'
            if isinstance(arg_info, dict):
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
        for arg, info in docstr_parser.unfinished_arguments.items():
            try:
                if name == "AFQ":
                    if '_params' in arg:
                        continue
                    section, desc = info['help'].split('[')[1].split(']')
                else:
                    section = name.upper()
                    desc = info['help']
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
