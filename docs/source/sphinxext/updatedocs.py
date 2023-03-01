# This updates usage/config.rst to the latest cli
# Developers can run this after modifying any arguments the user can see

from AFQ.utils.bin import func_dict_to_arg_dict, dict_to_toml

prologue = """
The pyAFQ configuration file
----------------------------

This file should be a `toml <https://github.com/toml-lang/toml>`_ file. At
minimum, the file should contain the BIDS path::

    [files]
    bids_path = "path/to/study"


But additional configuration options can be provided.
See an example configuration file below::

    title = "My AFQ analysis"

"""

epilogue = \
    """
pyAFQ will store a copy of the configuration file alongside the computed
results. Note that the `title` variable and `[metadata]` section are both for
users to enter any title/metadata they would like and pyAFQ will generally
ignore them.
"""


def setup(app):
    arg_dict = func_dict_to_arg_dict()
    example_config = dict_to_toml(arg_dict)
    example_config = '    ' + example_config.replace('\n', '\n    ')

    with open('./source/reference/config.rst', 'w') as ff:
        ff.write(prologue + example_config + epilogue)
