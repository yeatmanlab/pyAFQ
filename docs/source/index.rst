.. _home:

.. include:: <isonum.txt>

Automated Fiber Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tractography based on diffusion weighted MRI (dMRI) is used to find the major
white matter fascicles (tracts) in the living human brain. The health of these
tracts is an important factor underlying many cognitive and neurological
disorders.

`AFQ` is a software package focused on automated delineation of the major fiber
tracts in individual human brains, and quantification of the tissue properties
within the tracts.

.. todo::
  create and link to page that lists the `major fiber tracts` supported by
  default. include information on how to add custom fiber bundles defintions.

.. todo::
  create and link to page that identifies the supported `tissue properties`
  (believe this includes both dti and dki? fa, md, cfa?, and pdd?)

Tissue properties may vary systematically along each tract: different
populations of axons enter and exit the tract, and disease can strike at local
positions within the tract. Because of this, quantifying and understanding
diffusion measures along each fiber tract (the tract profile) may reveal new
insights into white matter development, function, and disease that are not
obvious from mean measures of that tract ([Yeatman2012]_).

.. [Yeatman2012] Jason D Yeatman, Robert F Dougherty, Nathaniel J Myall, Brian A Wandell, Heidi M Feldman, "Tract profiles of white matter properties: automating fiber-tract quantification", PloS One, 7: e49790


Tractometry
~~~~~~~~~~~

.. todo::
  include tractometry pipeline process flow diagram. create and link detail
  paged that elaborates on the process and each step within.

.. list-table:: AFQ Tractometry Pipeline
   :widths: auto

   * - Tractography
     - |rArr| `Registration <http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/registration/index.html>`_
     - |rArr| `Segmentation <http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/segmentation/index.html>`_
     - |rArr| `Cleaning <http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/segmentation/index.html#AFQ.segmentation.clean_bundles>`_
     - |rArr| `Profiling <https://dipy.org/documentation/1.4.0./reference/dipy.stats/#afq-profile>`_
     - |rArr| `Visualization <file:///Users/arokem/source/pyAFQ/docs/build/html/autoapi/AFQ/viz/index.html>`_

.. todo::
  can make this table a pretty graphic

.. todo::
  include detailed page that distguished tractometry versus tractography
  and benefits


Acknowledgements
~~~~~~~~~~~~~~~~

Work on this software is supported through grant `1RF1MH121868-01 <https://projectreporter.nih.gov/project_info_details.cfm?aid=9886761&icde=46874320&ddparam=&ddvalue=&ddsub=&cr=2&csb=default&cs=ASC&pball=>`_ from the `National Institutes for Mental Health <https://www.nimh.nih.gov/index.shtml>`_ / `The BRAIN Initiative <https://braininitiative.nih.gov>`_
and by a grant from the
`Gordon & Betty Moore Foundation <https://www.moore.org/>`_,  and from the
`Alfred P. Sloan Foundation <http://www.sloan.org/>`_ to the
`University of Washington eScience Institute <http://escience.washington.edu/>`_, by a `CRCNS <https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=5147>`_ grant (NIH
R01EB027585) to Eleftherios Garyfallidis and to Ariel Rokem , and by `NSF grant 1551330 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1551330>`_ to Jason Yeatman.

.. figure:: _static/eScience_Logo_HR.png
   :align: center
   :figclass: align-center
   :target: http://escience.washington.edu

.. figure:: _static/BDE_Banner_revised20160211-01.jpg
   :align: center
   :figclass: align-center
   :target: http://brainandeducation.com

.. toctree::
    :maxdepth: 2

    installation_guide
    usage/index
    auto_examples/index
    getting_help
    contributing
    autoapi/index
    developing/index
