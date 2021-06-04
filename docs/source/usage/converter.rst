Tractography from other pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyAFQ can use tractography from other pipelines. To tell pyAFQ to use
tractography from another pipeline, use the custom_tractography_bids_filters
argument in the AFQ object or in the configuration file. This argument expects
a dictionary of BIDS filters. pyAFQ will use these BIDS filters to find the
tractography for each subject in each session.
Here is an example custom_tractography_bids_filters::
    custom_tractography_bids_filters = {'scope': 'qsiprep', 'suffix': 'tractography'}
This would look for a file with the suffix 'tractography' inside of the
'qsiprep' pipeline. The name of the pipeline (in this case, 'qsiprep') should be
specified in its dataset_description.json.

There is an example usage of custom_tractography_bids_filters with the
:class:`AFQ.api.AFQ` object in the 'How pyAFQ uses BIDS' example in :ref:`examples`.


Matlab AFQ to Python AFQ conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyAFQ provides tools to convert between the results of the original Matlab
based AFQ (which we call mAFQ) and pyAFQ at various stages in the pipeline.

To convert an mAFQ tractography file, use :func:`AFQ.utils.conversion.matlab_tractography`. This function
takes in a path to the mAFQ tractography file and a path to an image.
The function returns a DIPY :class:`StatefulTractogram` class instance using the image as reference.
Here is an example::
    from AFQ.utils.conversion import matlab_tractography
    sft = matlab_tractography("WholeBrainFG-1106.mat", "sub-1106.dwi.nii.gz")

To convert an mAFQ mori groups file, use :func:`AFQ.utils.conversion.matlab_mori_groups`. This function
takes in a path to the mAFQ Mori groups file and a path to an image.
The function returns a dictionary where keys are the pyAFQ bundle names and
values are DIPY :class:`StatefulTractogram` class instances using the image as reference. This is
the same structure of the output of pyAFQ's segmentation.
Here is an example::
    from AFQ.utils.conversion import matlab_mori_groups
    fg = matlab_mori_groups("MoriGroups-1106.mat", "sub-1106.dwi.nii.gz")
If one wants to
convert this 'fg' dictionary to a single :class:`StatefulTractogram` object,
use :func:`bundles_to_tgram`.

Matlab tract profiles in the .csv format should already be compatible
with pyAFQ, so there is no need for conversion.