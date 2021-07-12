Adding Definitions 
~~~~~~~~~~~~~~~~~~
pyAFQ uses definitions to allow users to customize the mappings, masks,
etc. used by the :class:`AFQ.api.AFQ` object. All definitions ultimately
inherit from :class:`AFQ.definitions.utils.Definition`. For a given subject and
session within the API, a definition is used to create a given mask, map, etc.
`Definition`-inherited classes must have three methods: `__init__`, `find_path`, and `get_for_subses`.
The requirements of each of these methods are described below:

- `Definition`-inherited classes have an `__init__` method which the users uses to configure
  the the definition for a given instance of the :class:`AFQ.api.AFQ` object.
  These `__init__` methods must be thoroughly
  documented as they are what the user interacts with. The class must have attributes
  of same name as the `__init__` args. This is important for reading/writing
  `Definition`-inherited classes as arguments to config files.
  For scalar definitions, each class should have a `name` parameter (see
  :class:`AFQ.definitions.scalar.ScalarFile` source for an example). 

- The api calls `find_path` during the :class:`AFQ.api.AFQ` object initialization to
  let the definition find relevant files for the given subject and session. All `find_path`
  methods have the same input: `bids_layout`, `from_path`, `subject`, `session`. See
  :func:`AFQ.definitions.mask.MaskFile.find_path` for a basic example.
  If your definition does not need to run anything during :class:`AFQ.api.AFQ`
  object initialization, simply override this method with a method that only
  passes (see :func:`AFQ.definitions.mask.FullMask.find_path` source for an example.)

- The :class:`AFQ.api.AFQ` object calls `get_for_subses` to get the mask, map,
  etc. during workflow construction or execution. The form of this method varies significantly
  between mapping, scalar, and mask `Definition`-inherited classes. In
  mask `Definition`-inherited classes, `get_mask_getter` takes two arguments:
  `self` and `in_data`. The `in_data` argument specifies whether this mask is being
  called from the data Pimms plan (i.e., the brain mask) or not (i.e., the tractography
  seed and stop masks). `get_for_subses` then must return a task which can be called
  to generate the mask. In scalar `Definition`-inherited classes, `get_for_subses` only takes the `self`
  argument and similarly returns a task which generates the mask. In mapping
  `Definition`-inherited classes, `get_for_subses` should return a "mapping". The "mapping" must have
  `transform` and `transform_inverse` functions which each accept two arguments: (1)
  an ndarray called `data` and (2) `**kwargs`. In mapping `Definition`-inherited classes, get_for_subses
  takes as input self, subses_dict, reg_template, and reg_subject.

