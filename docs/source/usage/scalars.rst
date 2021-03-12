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
