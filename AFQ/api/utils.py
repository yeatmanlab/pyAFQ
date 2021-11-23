from importlib import import_module
from textwrap import dedent
import pimms
from funcargparse import FuncArgParser

__all__ = [
    "methods_descriptors", "kwargs_descriptors",
    "wf_sections", "add_method_descriptions", "AFQclass_doc"]

task_modules = ["data", "mapping", "segmentation", "tractography", "viz"]

methods_descriptors = {}
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
                for i in range(len(calc_obj.efferents)):
                    if eff_descs[i][0] == ' ' or eff_descs[i][0] == '\n':
                        eff_descs[i] = eff_descs[i][1:]
                    if eff_descs[i][0:3] == "and":
                        eff_descs[i] = eff_descs[i][3:]
                    if eff_descs[i][0] == ' ' or eff_descs[i][0] == '\n':
                        eff_descs[i] = eff_descs[i][1:]
                    methods_descriptors[
                        calc_obj.efferents[i]] =\
                        eff_descs[i]
            else:
                methods_descriptors[
                    calc_obj.efferents[0]] =\
                    docstr_parser.description
            for arg, info in docstr_parser.unfinished_arguments.items():
                if "help" in info:
                    if "default" in info:
                        default = info["default"]
                    else:
                        default = None
                    kwargs_descriptors[task_module][arg] = dict(
                        desc=info["help"],
                        kind=info["metavar"],
                        default=default)

# define sections in workflow dictionary
wf_sections = [
    "data_imap", "mapping_imap",
    "tractography_imap", "segmentation_imap",
    "subses_dict"]

AFQclass_doc = ""
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


def add_method_descriptions(AFQclass):
    # iterate through all attributes, setting methods for each one
    for output, desc in methods_descriptors.items():
        desc = desc.replace("\n", " ").replace("\t", "").replace("    ", "")
        exec(dedent(f"""\
        def export_{output}(self):
            \"\"\"
            Triggers a cascade of calculations to generate the desired output.
            Returns
            -------
            {desc}
            \"\"\"
            return self.{output}"""))
        fn = locals()[f"export_{output}"]
        if output[-5:] == "_file":
            setattr(AFQclass, f"export_{output[:-5]}", fn)
        else:
            setattr(AFQclass, f"export_{output}", fn)

    AFQclass_doc_intro = (
        "Here are the arguments you can pass to kwargs,"
        " to customize the tractometry pipeline. They are organized"
        " into 5 sections.\n")

    setattr(AFQclass, "__doc__", AFQclass_doc_intro + AFQclass_doc)

    return AFQclass
