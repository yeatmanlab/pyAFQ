Using pyAFQ
===========

Organizing your data
~~~~~~~~~~~~~~~~~~~~

The pyAFQ software assumes that your data is organized according to a format
that emulates the `BIDS <http://bids.neuroimaging.io/>`_ format.

Preprocessed diffusion MRI data, as well as anatomical and segmentation data
should be organized as follows:

|    study
|      ├-derivatives
|            ├-preafq
|                ├── sub01
|                │   ├── sess01
|                │   │   ├── anat
|                │   │   │   ├── sub-01_sess-01_aparc+aseg.nii.gz
|                │   │   │   └── sub-01_sess-01_T1.nii.gz
|                │   │   └── dwi
|                │   │       ├── sub-01_sess-01_dwi.bvals
|                │   │       ├── sub-01_sess-01_dwi.bvecs
|                │   │       └── sub-01_sess-01_dwi.nii.gz
|                │   └── sess02
|                │       ├── anat
|                │       │   ├── sub-01_sess-02_aparc+aseg.nii.gz
|                │       │   └── sub-01_sess-02_T1w.nii.gz
|                │       └── dwi
|                │           ├── sub-01_sess-02_dwi.bvals
|                │           ├── sub-01_sess-02_dwi.bvecs
|                │           └── sub-01_sess-02_dwi.nii.gz
|                └── sub02
|                   ├── sess01
|                   │   ├── anat
|                   │       ├── sub-02_sess-01_aparc+aseg.nii.gz
|                   │   │   └── sub-02_sess-01_T1w.nii.gz
|                   │   └── dwi
|                   │       ├── sub-02_sess-01_dwi.bvals
|                   │       ├── sub-02_sess-01_dwi.bvecs
|                   │       └── sub-02_sess-01_dwi.nii.gz
|                   └── sess02
|                       ├── anat
|                       │   ├── sub-02_sess-02_aparc+aseg.nii.gz
|                       │   └── sub-02_sess-02_T1w.nii.gz
|                       └── dwi
|                           ├── sub-02_sess-02_dwi.bvals
|                           ├── sub-02_sess-02_dwi.bvecs
|                           └── sub-02_sess-02_dwi.nii.gz


Where the `raw` directory contains your raw data, and the `preafq` directory
contains data that has been processed to be motion-corrected, eddy-current
corrected and so on. Anatomical data is optional. If a T1-weighted anatomical
image and its segmentation are not provided (i.e., there is no `anat` folder),
the software will use the diffusion data to estimate the parts of the image that
comprise the white matter.

.. note::

    The structure within the `raw` directory doesn't matter much to the `pyAFQ`
    software, as the software will not touch this data at all.

As part of its operation, the software will create another directory, under
`study/derivatives/afq`, which will contain all of the derivatives created by
the software.

.. note::

    As the BIDS format for derivatives matures, we intend to update the pyAFQ
    software to support a fully BIDS-compliant set of derivatives. Both as
    inputs, as well as outputs.


Running the pyAFQ pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

`pyAFQ` provides a programmatic application programming interface that can
be used to integrate pyAFQ functionality into Python programs or into
interactive analysis sessions (e.g., using
`Jupyter <https://jupyter.org>`_ notebooks). Examples of using the API are
provided in the :ref:`examples` documentation section.

In addition, pyAFQ provides a command-line interface. After installing the
software, and organizing the data, run::

    pyAFQ <preafq_path>

pointing the program to the location of the `preafq` folder. This will
run whole-brain tractography, segment the tracts, and extract tract-profiles
for each tract, generating a CSV file under
`study/derivatives/afq/tract_profiles.csv` that contains the tract profiles
for all participants/tracts/statistics

The CLI has several additional optional inputs::

  -s SUB_PREFIX, --sub_prefix SUB_PREFIX
                        Subject prefix (default: 'sub')
  -d DWI_FOLDER, --dwi_folder DWI_FOLDER
                        DWI folder (default: 'dwi')
  -f DWI_FILE, --dwi_file DWI_FILE
                        DWI file pattern (default: '*dwi')
  -a ANAT_FOLDER, --anat_folder ANAT_FOLDER
                        Anat folder (default: 'anat')
  -t ANAT_FILE, --anat_file ANAT_FILE
                        Anat file pattern (default: '*T1w*')
  -g SEG_FILE, --seg_file SEG_FILE
                        Segmentation file patter (default: '*aparc+aseg*')
  -b B0_THRESHOLD, --b0_threshold B0_THRESHOLD
                        B0 threshold (default: 0)
  -o ODF_MODEL, --odf_model ODF_MODEL
                        ODF model (default: 'DTI')
  -r DIRECTIONS, --directions DIRECTIONS
                        Tractography method (default: 'det')
  -n N_SEEDS, --n_seeds N_SEEDS
                        Number of seeds (default: 2 per voxel)
  -m, --random_seeds    Whether to use a total of `n_seeds` random seeds
                        instead of `n_seeds` per voxel (default: False)
