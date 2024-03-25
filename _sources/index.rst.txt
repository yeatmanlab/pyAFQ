.. _home:


Automated Fiber Quantification in Python (pyAFQ)
-------------------------------------------------

pyAFQ is an open-source software tool for the analysis of brain white matter in
diffusion MRI measurements. It implements a complete and automated data
processing pipeline for tractometry, from raw DTI data to white matter tract
identification, as well as quantification of tissue properties along the length
of the major long-range brain white matter connections.

- To get started, please refer to the `getting started <tutorials/index.html>`_ page.
- What is the difference between tractography and tractometry? See in the `explanations <explanations/index.html>`_ page.
- For more detailed information on the variety of uses of pyAFQ, see the `how to <howto/index.html>`_ page.
- For a detailed description of the methods and objects used in pyAFQ, see the `reference documentation <reference/index.html>`_ page.

.. todo::
  create and link to page that lists the `major fiber tracts` supported by
  default. include information on how to add custom fiber bundles defintions.

.. todo::
  create and link to page that identifies the supported `tissue properties`
  (believe this includes both dti and dki? fa, md, cfa?, and pdd?)

.. toctree::
    :maxdepth: 2
    :hidden:

    tutorials/index
    howto/index
    explanations/index
    reference/index


.. grid:: 2

    .. grid-item-card::
        :link: tutorials/index.html

        :octicon:`book;3em;sd-text-center`

        Tutorials
        ^^^^^^^^^

        Beginner's guide to pyAFQ. This guide introduces pyAFQ'S
        basic concepts and walks through fundamentals of using the software.

        +++

    .. grid-item-card::
        :link: howto/index.html

        :octicon:`rocket;3em;sd-text-center`

        How To
        ^^^^^^

        User's guide to pyAFQ. This guide assumes you know
        the basics and walks through some other commonly used functionality.

        +++

    .. grid-item-card::
        :link: explanations/index.html

        :octicon:`comment-discussion;3em;sd-text-center`

        Explanations
        ^^^^^^^^^^^^

        This guide contains in depth explanations of the various pyAFQ methods.

        +++

    .. grid-item-card::
        :link: reference/index.html

        :octicon:`search;3em;sd-text-center`

        API Reference
        ^^^^^^^^^^^^^

        The API Reference contains technical descriptions of methods
        and objects used in pyAFQ. It also contains descriptions
        of how methods work and the parameters used for each method.

        +++


Acknowledgements
~~~~~~~~~~~~~~~~

Work on this software was supported through grant `1RF1MH121868-01 <https://projectreporter.nih.gov/project_info_details.cfm?aid=9886761&icde=46874320&ddparam=&ddvalue=&ddsub=&cr=2&csb=default&cs=ASC&pball=>`_ from the `National Institutes for Mental Health <https://www.nimh.nih.gov/index.shtml>`_ / `The BRAIN Initiative <https://braininitiative.nih.gov>`_
and by a grant from the
`Gordon & Betty Moore Foundation <https://www.moore.org/>`_,  and from the
`Alfred P. Sloan Foundation <http://www.sloan.org/>`_ to the
`University of Washington eScience Institute <http://escience.washington.edu/>`_, by grant `R01EB027585 <https://reporter.nih.gov/search/jnnzzQ8Rj0CLD3R3l92GPg/project-details/10735068>`_ to Eleftherios Garyfallidis (PI) and Ariel Rokem, grant `R01HD095861 <https://reporter.nih.gov/search/j2JXd89oR0i4cCnIDo7fFA/project-details/10669103>`_ to Jason Yeatman, `R21HD092771 <https://reporter.nih.gov/search/j2JXd89oR0i4cCnIDo7fFA/project-details/9735358>`_  to Jason Yeatman and Pat Kuhl, by NSF grants `1551330 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1551330>`_ to Jason Yeatman and `1934292 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1934292>`_ to Magda Balazinska (PI) and Ariel Rokem (co-PI). John Kruper's work on pyAFQ has been supported through the NSF Graduate Research Fellowship program (DGE-2140004).


.. figure:: _static/eScience_Logo_HR.png
   :align: center
   :figclass: align-center
   :target: http://escience.washington.edu

.. figure:: _static/BDE_Banner_revised20160211-01.jpg
   :align: center
   :figclass: align-center
   :target: http://brainandeducation.com
