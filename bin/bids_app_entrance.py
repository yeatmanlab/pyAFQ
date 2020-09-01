
from argparse import ArgumentParser
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    print("Loading AFQ libraries...")
    from AFQ.api import AFQ
    from AFQ.utils.bin import func_dict_to_arg_dict, dict_to_toml, toml_to_val

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('pyAFQ')

special_args_dict = {
    "CLEANING": "clean_params",
    "SEGMENTATION": "segmentation_params",
    "TRACTOGRAPHY": "tracking_params"}

required_args = ['input_dataset', 'output_folder', 'analysis_level']

arg_dict = func_dict_to_arg_dict()
parser = ArgumentParser(description='pyAFQ BIDS App entrypoint script. ' + arg_dict["AFQ_desc"])

for section, args in arg_dict.items():
    if section == "AFQ_desc":
        continue
    for arg, arg_info in args.items():
        if section in special_args_dict.keys():
            arg = section.lower() + "__" + arg
        if arg in required_args:
            parser.add_argument(
                arg,
                help=arg_info['desc'])
        else:
            parser.add_argument(
                '--' + arg,
                dest=arg,
                default=arg_info['default'],
                help=arg_info['desc'])
parser.add_argument('analysis_level', help="This is ignored.")

args = parser.parse_args()

kwargs = {}
for arg, val in args.__dict__.items():
    is_special = False
    if '__' in arg:
        section, arg = arg.split('__')
        section = special_args_dict[section.upper()]
        if section not in kwargs:
            kwargs[section] = {}
        is_special = True

    if arg in required_args:
        continue

    val = toml_to_val(val)
    if is_special:
        kwargs[section][arg] = val
    else:
        kwargs[arg] = val

AFQ(args.input_dataset, output_folder=args.output_folder, **kwargs)
