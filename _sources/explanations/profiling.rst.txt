.. _profiling:

Profiling
=========
Once the bundles are identified, pyAFQ quantifies various properties of the
tract along its length. By default, these are fractional anisotropy and
mean diffusivity from diffusion tensor imaging. However, custom tissue
properties may be provided (see :ref:`custom-scalars` for details)
and there are a variety of other built-in tissue properties that you
can find in :ref:`methods_docs`.

These tract profiles are exported as a CSV. For each subject, 1-dimensional
profiles are generated for each tract and each tissue property. The CSV
can then be used for further statistical analysis. A useful library here 
is `AFQ-Insight <https://github.com/richford/AFQ-Insight>`_ .
