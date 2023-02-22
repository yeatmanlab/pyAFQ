Tractometry Pipeline
~~~~~~~~~~~~~~~~~~~~
.. include:: <isonum.txt>
    
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
     - |rArr| `Visualization <http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/viz/index.html>`_


.. .. figure:: _static/tractRegistration.png
..    :align: left
..    :scale: 30%
..    :figclass: align-center
..    :target: http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/registration/index.html

.. .. figure:: _static/tractSegmentation.png
..    :align: center
..    :scale: 30%
..    :figclass: align-center
..    :target: http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/segmentation/index.html

.. .. figure:: _static/tractCleaning.png
..    :align: right
..    :scale: 30%
..    :figclass: align-center
..    :target: http://yeatmanlab.github.io/pyAFQ/autoapi/AFQ/segmentation/index.html#AFQ.segmentation.clean_bundles

.. .. figure:: _static/tractProfiling.png
..    :align: left
..    :scale: 30%
..    :figclass: align-center
..    :target: https://dipy.org/documentation/1.4.0./reference/dipy.stats/#afq-profile

.. .. figure:: _static/tractVisualization.png
..    :align: center
..    :scale: 30%
..    :figclass: align-center
..    :target: file:///Users/arokem/source/pyAFQ/docs/build/html/autoapi/AFQ/viz/index.html