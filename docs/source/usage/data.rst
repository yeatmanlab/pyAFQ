Organizing your data
~~~~~~~~~~~~~~~~~~~~

pyAFQ works with `BIDS compliant <http://bids.neuroimaging.io/>`_ diffusion data.
While not required it is the prefered data format for neuroimaging interoperability.
Anatomical data and segmentation are optional. If a T1-weighted anatomical image and its
segmentation are not provided, the software will use the diffusion data to
estimate the parts of the image that comprise the white matter.

.. note::

    The structure within the `raw` part of the BIDS directory doesn't matter
    much to the `pyAFQ` software, as the software will not touch this data at
    all.

As part of its operation, the software will create another directory, under
`study/derivatives/afq`, which will contain all of the derivatives created by
the software.

.. note::

    As the BIDS format for derivatives matures, we intend to update the pyAFQ
    software to support a fully BIDS-compliant set of derivatives. Both as
    inputs, as well as outputs.

.. note::

    We rely heavily on `DIPY <https://dipy.org>`_  for many of the underlying
    implementational details. DIPY I/O functionality enforces that tractography
    results be saved in the RASMM space (see
    `this example <https://dipy.org/documentation/1.0.0./examples_built/streamline_formats/>`_
    for more explanation). This means that to transform the coordinates of
    streamlines stored in these files back into the subject-specific DWI space,
    they would have to be transformed through the affine stored in the header
    of the DWI nifti file.


Example data
------------

To get some example diffusion data formatted for input into `pyAFQ`, you can
run::

    import AFQ.data as afd
    afd.organize_stanford_data()

This should create a folder in your home directory with a properly-formatted
data set in a directory called `stanford_hardi`. Only the preprocessed
diffusion data is required for pyAFQ::

    ~/AFQ_data/stanford_hardi/derivatives/vistasoft/
    ├── dataset_description.json
    └── sub-01
        └── ses-01
            └── dwi
                ├── sub-01_ses-01_dwi.bval
                ├── sub-01_ses-01_dwi.bvec
                └── sub-01_ses-01_dwi.nii.gz

See :ref:`examples/plot_bids_layout` for a more extensive example.


Preprocessing
-------------
pyAFQ does not perform preprocessing. Instead, pyAFQ expects the outputs of
a preprocessing pipeline in BIDS format. In the above example, the dataset
was preprocessed using the `VISTASOFT <https://github.com/vistalab/vistasoft>`_ software.
Other examples may use other common preprocessing
software tools, such as `dMRIPrep <https://github.com/nipreps/dmriprep>`_
or `QSIprep <https://qsiprep.readthedocs.io/en/latest/>`_.

.. note::

    The outputs of VISTASOFT are stored in its own pipeline folder, which we
    chose to call `derivatives/vistasoft`. In general, any folder name is
    valid, as long as the folder is inside of the `derivatives` folder.

.. note::

    By default, pyAFQ will check all folders in `derivatives` for data.
    If you want to specify which pipeline pyAFQ should check for preprocessed
    data, pass that pipeline's name to the `dmriprep` parameter.
    The name of the pipeline is specified in the dataset_description.json,
    it is not based on the folder name.
