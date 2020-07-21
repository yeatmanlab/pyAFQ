#!/usr/bin/env python

import warnings
import os.path as op

from funcargparse import FuncArgParser
from gooey import Gooey
from configupdater import ConfigUpdater

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('pyAFQ')
    logger.info("Loading AFQ api...")

    from AFQ.api import AFQ
    from AFQ.utils.bin import get_default_args
    import AFQ.segmentation as seg
    import AFQ.tractography as aft

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


track_defaults = get_default_args(aft.track)
track_defaults_list = process_defaults(track_defaults)
seg_defaults = get_default_args(seg.Segmentation.__init__)
seg_defaults_list = process_defaults(seg_defaults)
clean_defaults = get_default_args(seg.clean_bundle)
clean_defaults_list = process_defaults(clean_defaults)

@Gooey
def parse():
    parser = FuncArgParser()
    parser.setup_args(AFQ.__init__)

    parser.append2help(
        'segmentation_params',
        " Parameters Include: \n\t" + "\n\t".join(seg_defaults_list))
    parser.append2help(
        'tracking_params',
        " Parameters Include: \n\t" + "\n\t".join(track_defaults_list))
    parser.append2help(
        'clean_params',
        " Parameters Include: \n\t" + "\n\t".join(clean_defaults_list))

    config_file = "./test.ini"
    cfparser = ConfigUpdater()
    if not op.exists(config_file):
        open(config_file, 'w').close()
    cfparser.read(config_file)

    # updates any arguments not currently in .ini
    for arg, info in parser.unfinished_arguments.items():
        try:
            section, desc = info['help'].split('[')[1].split(']')
        except KeyError:
            logger.warning(
                "You are missing a valid description for the "
                + "AFQ argument " + arg)
            continue
        if cfparser.has_option(section, arg):
            new_default = cfparser[section][arg]
            if new_default is "''":
                new_default = None
            parser.unfinished_arguments[arg]['default'] = new_default
                
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
    parser.create_arguments()
    # TODO: cli, gui

if __name__ == '__main__':
    parse()