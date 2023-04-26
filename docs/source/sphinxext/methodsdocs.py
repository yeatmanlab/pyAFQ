# This updates usage/methods.rst to the latest API methods
# Developers can run this after modifying any arguments the user can see

from AFQ.api.utils import methods_descriptors
from textwrap import dedent

prologue = """
.. _methods_docs:

The pyAFQ API methods
---------------------
After defining your pyAFQ API object, you can ask for the output of
any step of the pipeline. It is common for users to just call export_all
(for example, `myafq.export_all()`). However, if the user only wants the
tractography, the user can instead call `myafq.export("streamlines")`. Here
is a list of all of pyAFQ's possible outputs:

"""


def setup(app):
    method_descriptions = ""
    for output, desc in methods_descriptors.items():
        desc = desc.replace("\n", " ").replace("\t", "").replace("    ", "")
        method_descriptions = method_descriptions + dedent(f"""

        {output}:
            {desc}
        """)

    with open('./source/reference/methods.rst', 'w') as ff:
        ff.write(prologue + method_descriptions)
