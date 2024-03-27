Bundle Orientation
~~~~~~~~~~~~~~~~~~
The result of tractometry is a tract profile, which describe how tissue
properties vary along the length of a tract. These profiles are discretized,
and represented as 100 tissue property samples equally spaced along the length of
the tract, from the start to the end. This raises the question of how tracts are
oriented.

In pyAFQ, the orientation is defined by the bundle dictionary, see:
:ref:`bundle-dict-label`. Tracts start at the start ROI,
go through the inclusion ROIs in the order they are
provided, and then end in the end ROI. Of course, each of these ROIs is not
required for each bundle. At minimum, a bundle must have either a start or end
ROI, or at least two inclusion ROIs, so the orientation can be defined.
This also means that the order of the inclusion ROIs must be the same as the
order of the start and end ROIs in the bundle dictionary.

Most users can ignore these nuances and use the default bundle dictionaries
provided by pyAFQ. For all of our default bundles, we follow the 
LPI+ convention. This means tract profiles go from right to left, from
anterior to posterior, or from superior to inferior, depending on the primary
orientation of the bundle. Note that this does not mean the TRK/TRX files are
saved using the LPI+ convention. Streamlines coordinates are saved in the
standard RAS orientation, it is just that tract profiles are in LPI+
orientation. For some bundles, such as the inferior longitudinal
fasciculus, the choice of primary orientation could be unclear. So, here
we list the orientations of the standard bundles::
 - Left Anterior Thalamic: anterior to posterior
 - Right Anterior Thalamic: anterior to posterior
 - Left Cingulum Cingulate: anterior to posterior
 - Right Cingulum Cingulate: anterior to posterior
 - Left Corticospinal: superior to inferior
 - Right Corticospinal: superior to inferior
 - Left Inferior Fronto-occipital: anterior to posterior
 - Right Inferior Fronto-occipital: anterior to posterior
 - Left Inferior Longitudinal: anterior to posterior
 - Right Inferior Longitudinal: anterior to posterior
 - Left Superior Longitudinal: anterior to posterior
 - Right Superior Longitudinal: anterior to posterior
 - Left Arcuate: anterior to posterior
 - Right Arcuate: anterior to posterior
 - Left Uncinate: superior to inferior
 - Right Uncinate: superior to inferior
 - Forceps Minor: right to left
 - Forceps Major: right to left
 - Left Posterior Arcuate: superior to inferior
 - Right Posterior Arcuate: superior to inferior
 - Left Vertical Occipital: superior to inferior
 - Right Vertical Occipital: superior to inferior

All callosal bundles are oriented from right to left, and all other
custom bundles provided in the examples follow the LPI+ convention.
