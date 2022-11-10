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

For an example, see "Plotting the Optic Radiations" in :ref:`examples`.
