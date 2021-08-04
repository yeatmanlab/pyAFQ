The pyAFQ Custom Scalars API
~~~~~~~~~~~~~~~~~~
pyAFQ has a system for users to add custom scalars (scalars beyond the several
we calculate by default). The scalars API is similar to our Mask API.

In AFQ/definitions/scalar.py, there are two scalar classes one
can use to specify custom scalars. As a user, one should initialize scalar
classes and pass them to the AFQ object, or write out the initialization as
a string inside of one's configuration file for use with the CLI. To do this,
give a scalar object as an element of the scalars array passed to :class:`AFQ.api.AFQ`.
Then your custom scalar will be automatically used during tract profile extraction.

- :class:`AFQ.definitions.mask.TemplateMask`: This class can be used if you want to transform a scalar
  you made in some template space to each subject space before using it.

- :class:`AFQ.definitions.mask.ScalarFile`: This class can be used if you have your scalar in subject
  space, and there is a scalar file in BIDS format already for each subject.

As an example, one might have T1 files in a BIDS pipeline named "freesurfer"
and with the suffix "T1w" for each subject and session::
  
      ~/my_bids_path/
      ├── dataset_description.json
      └── derivatives
          ├── freesurfer
          │   ├── dataset_description.json
          │   └── sub-01
          │       └── ses-01
          │           └── anat
          │               └── sub-01_ses-01_T1w.nii.gz
          └── vistasoft
              ├── dataset_description.json
              └── sub-01
                  └── ses-01
                      └── dwi
                          ├── sub-01_ses-01_dwi.bval
                          ├── sub-01_ses-01_dwi.bvec
                          └── sub-01_ses-01_dwi.nii.gz

You can use :class:`AFQ.definitions.mask.ScalarFile`
to provide this custom scalar to the AFQ object::

    my_custom_t1_scalar = ScalarFile(
      "T1",
      "T1w",
      {"scope": "freesurfer"})

    api.AFQ("my_bids_path", scalars=["dti_fa", "dti_md", my_custom_t1_scalar])
