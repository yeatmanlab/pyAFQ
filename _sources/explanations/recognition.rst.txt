.. _recognition:

Bundle Recognition
==================
pyAFQ segments streamlines into different known fiber bundles,
like the corpus callosum, corticospinal tract, or uncinate fasciculus.
This is typically achieved through a combination of anatomical landmarks
(either in the core white matter, as inclusion/exclusion regions of interest,
or using startpoint/endpoint regions in the gray matter). Streamline
orientation, length, and curvature are also used to help identify bundles.
After initial segmentation, the bundle often undergoes cleaning to remove
spurious streamlines that likely do not belong to the target bundle.
You can see the :mod:`AFQ.segmentation` module for details.
