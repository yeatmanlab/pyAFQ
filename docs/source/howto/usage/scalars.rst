
How to add custom tissue properties from another pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per default, pyAFQ calculates tract profiles of the FA and MD using DTI
in every bundle that it recognizes. However, there is also a system for users
to add tissue properties or other scalar images (i.e., 3D images with one
numeric value per voxel).

In AFQ/definitions/image.py, there are many classes one can use to define
custom images. Two of these classes are particularly useful to
specify custom scalars. As a user, one should initialize one of these
classes and pass them to the AFQ.api objects, or write out the initialization as
a string inside of one's configuration file for use with the CLI. To do this,
give an image object as an element of the scalars array passed to :class:`AFQ.api.group.GroupAFQ`.
Then your custom image will be automatically used during tract profile extraction.

- :class:`AFQ.definitions.image.TemplateImage`: This class can be used if you want to transform a scalar
  you made in some template space to each subject space before using it.

- :class:`AFQ.definitions.image.ImageFile`: This class can be used if you have your scalar in subject
  space, and there is a scalar file in BIDS format already for each subject.

As an example, one might have "ICVF" and "ODI" maps in a BIDS pipeline named "noddi"::

      ~/my_bids_path/
      ├── dataset_description.json
      └── derivatives
          ├── noddi
          │   ├── dataset_description.json
          │   └── sub-01
          │       └── ses-01
          │           └── sub-01_ses-01_ICVF.nii.gz
          │           └── sub-01_ses-01_ODI.nii.gz
          └── vistasoft
              ├── dataset_description.json
              └── sub-01
                  └── ses-01
                      └── dwi
                          ├── sub-01_ses-01_dwi.bval
                          ├── sub-01_ses-01_dwi.bvec
                          └── sub-01_ses-01_dwi.nii.gz

You can use :class:`AFQ.definitions.image.ImageFile`
to provide these custom scalars to the AFQ.api objects::

    ICVF_scalar = ImageFile(
      suffix="ICVF",
      filters={"scope": "noddi"})

    ODI_scalar = ImageFile(
      suffix="ODI",
      filters={"scope": "noddi"})

    api.GroupAFQ(
      "my_bids_path",
      scalars=["dti_fa", "dti_md", ICVF_scalar, ODI_scalar])

Or provide them using the CLI, by adding them to the `scalars` parameter::

    scalars = ["dti_fa", "dti_md", "ImageFile(suffix='ODI', filters={'scope': 'noddi'})", "ImageFile(suffix='ICVF', filters={'scope': 'noddi')" ]
