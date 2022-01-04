pyAFQ has a system for defining custom bundles. Custom bundles are defined
by passing a custom bundle_info dicitonary to
:class:`AFQ.api.bundle_dict.BundleDict`: . The keys of bundle_info are bundle
names; the values are another dictionary describing the bundle, with these
key-value pairs::
    - 'include' : a list of Nifti1Images containing inclusion ROI(s)
    - 'exclude' : a list of Nifti1Images containing exclusion ROI(s), optional
    - 'start' : a Nifti1Image containing the start ROI, optional
    - 'end' : a Nifti1Image containing the end ROI, optional
    - 'cross_midline' : boolean describing whether the bundle crosses the
      midline, optional. If None, the bundle may or may not cross the midline.
    - 'space' : a string which is either 'template' or 'subject', optional
    If this field is not given or 'template' is given, the ROI will be
    transformed from template to subject space before being used.
    - 'prob_map' : a Nifti1Images which is the probability map, optional

For an example, see "Plotting the Optic Radiations" in :ref:`examples`.
