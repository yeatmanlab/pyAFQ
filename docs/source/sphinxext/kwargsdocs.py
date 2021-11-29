# This updates usage/config.rst to the latest cli
# Developers can run this after modifying any arguments the user can see

from AFQ.api.utils import methods_descriptors, AFQclass_doc
from textwrap import dedent

prologue = """
.. _kwargs_docs:

The pyAFQ API optional arguments
--------------------------------
You can run pyAFQ on either a subject or participant level
using pyAFQ's API objects, :class:`AFQ.api.group.GroupAFQ`
and :class:`AFQ.api.participant.ParticipantAFQ`. Either way,
these classes take additional optional arguments. These arguments
give the user control over each step of the tractometry pipeline,
allowing customizaiton of tractography, bundle recognition, registration,
etc. Here are all of these arguments and their descriptions, organized
into 5 sections:

"""

middlogue = """

The pyAFQ API methods
---------------------
After defining your pyAFQ API object, you can ask for the output of
any step of the pipeline. It is common for users to just call export_all
(for example, `myafq.export_all()`). However, if the user only wants the
tractography, the user can instead call `myafq.export_tractography()`. Here
is a list of all of pyAFQ's possible outputs:

"""


def setup(app):
    method_descriptions = ""
    for output, desc in methods_descriptors.items():
        desc = desc.replace("\n", " ").replace("\t", "").replace("    ", "")
        method_descriptions = method_descriptions + dedent(f"""

        def  export_{output}(self):
            {desc}
        """)

    with open('./source/usage/kwargs.rst', 'w') as ff:
        ff.write(prologue + AFQclass_doc + middlogue + method_descriptions)
