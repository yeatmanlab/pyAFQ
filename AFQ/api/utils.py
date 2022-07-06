import contextlib
from importlib import import_module
from AFQ.viz.utils import viz_import_msg_error
import pimms
from funcargparse import FuncArgParser
import logging
import warnings

from dipy.io.stateful_tractogram import set_sft_logger_level


with contextlib.suppress(Exception):  # only works on python 3.9
    from outdated import OutdatedPackageWarning
    warnings.filterwarnings("ignore", category=OutdatedPackageWarning)


__all__ = [
    "methods_descriptors", "kwargs_descriptors", "AFQclass_doc"]


set_sft_logger_level(logging.CRITICAL)


task_modules = ["data", "mapping", "segmentation", "tractography", "viz"]

methods_descriptors = {}
methods_sections = {}
kwargs_descriptors = {}
for task_module in task_modules:
    kwargs_descriptors[task_module] = {}
    for calc_obj in import_module(
            f"AFQ.tasks.{task_module}").__dict__.values():
        if isinstance(calc_obj, pimms.calculation.Calc):
            docstr_parser = FuncArgParser()
            docstr_parser.setup_args(calc_obj.function)
            if len(calc_obj.efferents) > 1:
                eff_descs = docstr_parser.description.split(",")
                if len(eff_descs) != len(calc_obj.efferents):
                    raise NotImplementedError((
                        "If calc method has mutliple outputs, "
                        "their descriptions must be divided by commas."
                        f" {calc_obj.name} has {len(eff_descs)} comma-divided"
                        f"sections but {len(calc_obj.efferents)} outputs"))
                for ii in range(len(calc_obj.efferents)):
                    if eff_descs[ii][0] in [' ', '\n']:
                        eff_descs[ii] = eff_descs[ii][1:]
                    if eff_descs[ii][:3] == "and":
                        eff_descs[ii] = eff_descs[ii][3:]
                    if eff_descs[ii][0] in [' ', '\n']:
                        eff_descs[ii] = eff_descs[ii][1:]
                    methods_descriptors[
                        calc_obj.efferents[ii]] =\
                        eff_descs[ii]
                    methods_sections[calc_obj.efferents[ii]] =\
                        task_module
            else:
                methods_descriptors[
                    calc_obj.efferents[0]] =\
                    docstr_parser.description
                methods_sections[calc_obj.efferents[0]] =\
                    task_module
            for arg, info in docstr_parser.unfinished_arguments.items():
                if "help" in info:
                    default = info["default"] if "default" in info else None
                    kwargs_descriptors[task_module][arg] = dict(
                        desc=info["help"],
                        kind=info["metavar"],
                        default=default)
                if arg not in methods_sections:
                    methods_sections[arg] = task_modules[-1]


AFQclass_doc = (
    "Here are the arguments you can pass to kwargs,"
    " to customize the tractometry pipeline. They are organized"
    " into 5 sections.\n")
for task_module in task_modules:
    AFQclass_doc = AFQclass_doc + "\n"
    AFQclass_doc = AFQclass_doc +\
        "==========================================================\n"
    AFQclass_doc = AFQclass_doc + task_module.upper() + "\n"
    AFQclass_doc = AFQclass_doc +\
        "==========================================================\n"
    for arg, info in kwargs_descriptors[task_module].items():
        AFQclass_doc = AFQclass_doc + arg + ": " + info["kind"]
        AFQclass_doc = AFQclass_doc + "\n\t"
        AFQclass_doc = AFQclass_doc + info["desc"].replace(
            "\n", "\n\t")
        AFQclass_doc = AFQclass_doc + "\n\n"


valid_exports_string = (
    "Here is a list of valid attributes "
    f"to export: {methods_sections.keys()}")


def check_attribute(attr_name):
    if attr_name == "help":
        print(valid_exports_string)
        return None

    if attr_name in methods_sections:
        if methods_sections[attr_name] == task_modules[-1]:
            return None
        else:
            return f"{methods_sections[attr_name]}_imap"

    raise ValueError(
        f"{attr_name} not found for export. {valid_exports_string}")


def export_all_helper(api_afq_object, seg_algo, xforms, indiv, viz):
    if xforms:
        try:
            api_afq_object.export("b0_warped")
        except Exception as e:
            api_afq_object.logger.warning((
                "Failed to export warped b0. This could be because your "
                "mapping type is only compatible with transformation "
                f"from template to subject space. The error is: {e}"))
        api_afq_object.export("template_xform")

    if indiv:
        api_afq_object.export("indiv_bundles")
        if seg_algo == "AFQ":
            api_afq_object.export("rois")
    api_afq_object.export("sl_counts")
    api_afq_object.export("median_bundle_lengths")
    api_afq_object.export("profiles")

    if viz:
        try:
            api_afq_object.export("tract_profile_plots")
        except ImportError as e:
            plot_err_message = viz_import_msg_error("plot")
            if str(e) != plot_err_message:
                raise
            else:
                api_afq_object.logger.warning(plot_err_message)
        api_afq_object.export("all_bundles_figure")
        if seg_algo == "AFQ":
            api_afq_object.export("indiv_bundles_figures")
