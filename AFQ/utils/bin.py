
import inspect
from argparse import ArgumentParser


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
