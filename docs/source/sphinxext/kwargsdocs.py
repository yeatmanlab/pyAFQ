# This updates usage/kwargs.rst to the latest API kwargs
# Developers can run this after modifying any arguments the user can see

from AFQ.api.utils import AFQclass_doc

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


def setup(app):
    with open('./source/reference/kwargs.rst', 'w') as ff:
        ff.write(prologue + AFQclass_doc)
