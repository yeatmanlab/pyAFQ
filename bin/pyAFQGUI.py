#!/usr/bin/env pythonw

import warnings
import os.path as op

from funcargparse import FuncArgParser
from argparse import ArgumentParser, FileType
from cli2gui import Cli2Gui
from configupdater import ConfigUpdater

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('pyAFQ')
    logger.info("Loading AFQ api...")

    import AFQ
    import AFQ.api as api
    from AFQ.utils.bin import get_default_args
    import AFQ.segmentation as seg
    import AFQ.tractography as aft

use_config_gui = True
gui_theme = \
    ["#e7e7e9", "#dfdfe1", "#cacace", "#a0a1a7", "#696c77",
     "#383a42", "#202227", "#090a0b", "#ca1243", "#c18401", "#febb2a",
     "#50a14f", "#0184bc", "#4078f2", "#a626a4", "#986801", "#f0f0f1",
     "#fafafa", "#ec2258", "#f4a701", "#6db76c", "#01a7ef", "#709af5",
     "#d02fcd"]
gui_sizes = \
    {"title_size": 36,
     "label_size": (30, None),
     "input_size": (30, 1),
     "button": (10, 1),
     "padding": (5, 10),
     "helpText_size": 2,  # no effect?
     "text_size": 18},
usage = \
f"""pyAFQ /path/to/afq_config.ini

Runs full AFQ processing as specified in the configuration file.

To not run the CLI with a GUI, use --disable-cli2gui .

To not open a GUI for editing the config file, use --disable-config2gui .

For details about configuration, see instructions in:
https://yeatmanlab.github.io/pyAFQ/usage.html#running-the-pyafq-pipeline
"""


def process_defaults(defaults_dict):
    defaults_list = []
    for k, v in defaults_dict.items():
        if isinstance(v, str):
            defaults_list.append(f"{k} = '{v}'")
        elif isinstance(v, bool):
            if v:
                defaults_list.append(f"{k} = true")
            else:
                defaults_list.append(f"{k} = false")
        elif callable(v):
            defaults_list.append(f"{k} = '{v.__name__}'")
        else:
            defaults_list.append(f"{k} = {v}")
    return defaults_list


@Cli2Gui(
    program_name="pyAFQ CLI",
    program_description=usage,
    image=None,
    sizes=gui_sizes,
    auto_enable=True,
    gui="pysimplegui",
    parser="argparse",
    theme=gui_theme,
    darktheme=None)
def parse_cli():
    # cli_parser parses user arguments from CLI,
    # It is also what gooey uses to make the GUI
    cli_parser = ArgumentParser(usage=usage)

    cli_parser.add_argument(
        dest='config',
        action="store",
        type=FileType('w'),
        help="Path to config file. For example, /path/to/afq_config.ini")

    cli_parser.add_argument(
        '-t', '--notrack', action="store_true", default=False,
        help="Disable the use of pyAFQ being recorded by Google Analytics. ")
    
    cli_parser.add_argument(
        '--disable-config2gui', action="store_true", default=False,
        help="Disable the GUI for editing the config file. ")

    opts = cli_parser.parse_args()

    if not opts.notrack:
        logger.info(
            "Your use of pyAFQ is being recorded using Google Analytics. "
            "For more information, please refer to the pyAFQ documentation: "
            "https://yeatmanlab.github.io/pyAFQ/usage.html#usage-tracking-with-google-analytics. "  # noqa
            "To turn this off, use the `--notrack` flag when using the pyAFQ CLI")
        import popylar
        popylar.track_event(AFQ._ga_id, "pyAFQ_cli", "CLI called")
    opts.config.close()
    use_config_gui = not opts.disable_config2gui # TODO: make this work

    return opts.config.name


Cli2Gui(
    run_function=parse_doc,
    program_name="pyAFQ Configuration",
    program_description="Set parameters in config file.",
    image=None,
    sizes={
		"title_size": 36,
		"label_size": (30, None),
		"input_size": (30, 1),
		"button":(10, 1),
		"padding":(5, 10),
		"helpText_size": 2, # no effect?
		"text_size": 18},
    auto_enable=use_config_gui,
    gui="pysimplegui",
    parser="argparse",
    theme=gui_theme,
    darktheme=None)
def parse_doc():
    # doc_parser organizes the docstring for AFQ.api.AFQ
    docstr_parser = FuncArgParser()
    docstr_parser.setup_args(api.AFQ.__init__)

    track_defaults = get_default_args(aft.track)
    track_defaults_list = process_defaults(track_defaults)
    seg_defaults = get_default_args(seg.Segmentation.__init__)
    seg_defaults_list = process_defaults(seg_defaults)
    clean_defaults = get_default_args(seg.clean_bundle)
    clean_defaults_list = process_defaults(clean_defaults)

    docstr_parser.append2help(
        'segmentation_params',
        " Parameters Include: \n\t" + "\n\t".join(seg_defaults_list))
    docstr_parser.append2help(
        'tracking_params',
        " Parameters Include: \n\t" + "\n\t".join(track_defaults_list))
    docstr_parser.append2help(
        'clean_params',
        " Parameters Include: \n\t" + "\n\t".join(clean_defaults_list))

    docstr_parser.create_arguments()
    opts = docstr_parser.parse_args()

    return docstr_parser.unfinished_arguments, opts # TODO: or between opts and unfinished args


def parse_config(config_file, doc_args):
    # cfparser parses the config .ini file
    cfparser = ConfigUpdater()
    if not op.exists(config_file):
        open(config_file, 'w').close()
    cfparser.read(config_file)

    # updates any arguments not currently in .ini
    for arg, info in doc_args.items():
        try:
            section, desc = info['help'].split('[')[1].split(']')
        except KeyError:
            logger.warning(
                "We are missing a valid description for the "
                + "AFQ argument " + arg)
            continue
        if cfparser.has_option(section, arg):
            new_default = cfparser[section][arg]
            if new_default is "''":
                new_default = None
            doc_args[arg]['default'] = new_default

        else:
            if not cfparser.has_section(section):
                cfparser.add_section(section)
            if 'default' in info and info['default'] is not None:
                default = info['default']
            else:
                default = "''"
            cfparser[section][arg] = default

            for desc_line in desc.splitlines():
                cfparser[section][arg].add_before.comment(
                    desc_line, comment_prefix=';')

    cfparser.update_file()

    breakpoint()
    # TODO: from config, call AFQ


if __name__ == '__main__':
    config_file = parse_cli()
    doc_args = parse_doc()
    parse_config(config_file, doc_args)
