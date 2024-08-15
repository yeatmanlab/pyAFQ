.. _bundle-dict-label:

Defining Custom Bundle Dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyAFQ has a system for defining custom bundles. Custom bundles are defined
by passing a custom `bundle_info` dictionary to
:class:`AFQ.api.bundle_dict.BundleDict`: The keys of `bundle_info` are bundle
names; the values are another dictionary describing the bundle, with these
key-value pairs::
    - 'include' : a list of paths to Nifti files containing inclusion ROI(s).
      One must either have at least 1 include ROI, or 'start' or 'end' ROIs.
    - 'exclude' : a list of paths to Nifti files containing exclusion ROI(s),
      optional.
    - 'start' : path to a Nifti file containing the start ROI, optional
    - 'end' : path to a Nifti file containing the end ROI, optional
    - 'cross_midline' : boolean describing whether the bundle is required to
      cross the midline (True) or prohibited from crossing (False), optional.
      If None, the bundle may or may not cross the midline.
    - 'space' : a string which is either 'template' or 'subject', optional
    If this field is not given or 'template' is given, the ROI will be
    transformed from template to subject space before being used.
    - 'prob_map' : path to a Nifti file which is the probability map,
      optional.
    - 'inc_addtol' : List of floats describing how much tolerance to add or
      subtract in mm from each of the inclusion ROIs. The list must be the
      same length as 'include'. optional. 
    - 'exc_addtol' : List of floats describing how much tolerance to add or
      subtract in mm from each of the exclusion ROIs. The list must be the
      same length as 'exclude'. optional. 
    - 'mahal': Dict describing the parameters for cleaning. By default, we
      use the default behavior of the seg.clean_bundle function.
    - 'recobundles': Dict which should contain an 'sl' key and 'centroid'
      key. The 'sl' key should be the reference streamline and the 'centroid'
      key should be the centroid threshold for Recobundles.
    - 'qb_thresh': Float which is the threshold for Quickbundles cleaning.
    - 'primary_axis': int which is the index of the primary axis the
      bundle should travel in.
    - 'primary_axis_percentage': Used with primary_axis, defines what fraction
      of a streamlines movement should be in the primary axis.
    - 'length': dicitonary containing 'min_len' and 'max_len'


When doing bundle recognition, streamlines are filtered out from the whole
tractography according to the series of steps defined in the bundle
dictionaries. Of course, no one bundle uses every step, but here is the order
of the steps:
  1. Probability Maps
  2. Crosses midline
  3. Startpoint
  4. Endpoint
  5. Min and Max length
  6. Primary axis
  7. Include
  8. Curvature
  9. Exclude
  10. Recobundles
  11. Quickbundles Cleaning
  12. Mahalanobis Cleaning
If a streamline passes all steps for a bundle, it is included in that bundle.
If a streamline passess all steps for multiple bundles, then a warning is
thrown and the tie goes to whichever bundle is first in the bundle dictionary.


If, for debugging purposes, you want to save out the streamlines
remaining after each step, set `save_intermediates` to a path in
`segmentation_params`. Then the streamlines will be saved out after each step
to that path. Only do this for one subject at a time.


Custom bundle definitions such as the OR, and the standard BundleDict can be
combined through addition. For an example, see "Plotting the Optic Radiations" in :ref:`examples`.
Some tracts, such as the Vertical Occipital Fasciculus, may be defined relative
to other tracts. In those cases, the custom tract definitions should appear in the BundleDict 
object after the reference tracts have been defined. These reference tracts can 
be included as keys in the same dictionary for that tract. For example::

   newVOF = abd.BundleDict({
                'Left Vertical Occipital': {'cross_midline': False,
                                    'space': 'template',
                                    'start': templates['VOF_L_start'],
                                    'end': templates['VOF_L_end'],
                                    'inc_addtol': [4, 0],
                                    'Left Arcuate': {
                                        'node_thresh': 20},
                                    'Left Posterior Arcuate': {
                                        'node_thresh': 1,
                                        'core': 'Posterior'},
                                    'Left Inferior Longitudinal': {
                                        'core': 'Left'},
                                    'primary_axis': 2,
                                    'primary_axis_percentage': 40}
                            })

This definition of the VOF in the custom BundleDict would first require left ARC, left pARC, and left ILF 
to be defined, in the same way the tiebreaker above works. You would then construct your custom 
BundleDict like this. The order of addition matters here::

    BundleDictCustomVOF = abd.default18_bd() + newVOF
