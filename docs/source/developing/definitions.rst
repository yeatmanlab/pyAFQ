Adding Definitions 
~~~~~~~~~~~~~~~~~~
pyAFQ uses definitions to allow users to customize the mappings, masks,
etc. used by the :class:`AFQ.api.AFQ` object. All definitions ultimately
inherit from :class:`AFQ.definitions.utils.Definition`. For a given subject and
session within the API, a definition is used to create a given mask, map, etc.
Definitions must have three methods: `__init__`, `find_path`, and `get_for_subses`.
The requirements of each of these methods are described below:

- Definitions have an init method which the users uses to specify
  how they want the definition to behave. These init methods must be thoroughly
  documented as they are what the user interacts with. The class must have attributes
  of same name as the init args. This is important for reading/writing definitions
  as arguments to config files. For scalar defintions, each class should have a
  name parameter. 

- The api calls find_path during the :class:`AFQ.api.AFQ` object initialization to
  let the definition find relevant files for the given subject and session. All find_path
  methods have the same input: bids_layout, from_path, subject, session. See
  :func:`AFQ.definitions.mask.MaskFile.find_path` for a basic example.
  If your definition does not need to run anything during :class:`AFQ.api.AFQ`
  object initialization, simply override this method with a method that only
  passes (see :func:`AFQ.definitions.mask.FullMask.find_path` for an example.)

- The :class:`AFQ.api.AFQ` object calls get_for_subses to get the mask, map,
  etc. during workflow construction or execution. The form of thie method varies significantly
  between mapping, scalar, and mask definitions. In mask definitions, get_mask_getter
  takes two arguments: self and in_data. in_data specifies whether this mask is being
  called from the data Pimms plan (ie, the brain mask) or not (ie, the tractography
  seed and stop masks). get_for_subses then must return a task which can be called
  to generate the mask. In scalar definitions, get_for_subses only takes the self
  argument and similarly returns a task which generates the mask. In mapping
  definitions, get_for_subses should return a "mapping". The "mapping" must have
  transform and transform_inverse functions which each accept two arguments: (1)
  an ndarray called data and (2) **kwargs. In mapping definitions, get_for_subses
  takes as input self, subses_dict, reg_template, and reg_subject.

