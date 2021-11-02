from importlib import import_module

__all__ = ["task_outputs", "wf_sections"]

task_outputs = {}
for task_module in ["data", "mapping", "segmentation", "tractography", "viz"]:
    task_outputs.update(import_module(f"AFQ.tasks.{task_module}").outputs)

# define sections in workflow dictionary
wf_sections = [
    "data_imap", "mapping_imap",
    "tractography_imap", "segmentation_imap",
    "subses_dict"]
