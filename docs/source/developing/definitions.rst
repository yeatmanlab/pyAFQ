Adding Definitions 
~~~~~~~~~~~~~~~~~~
pyAFQ uses definitions to allow users to customize the mappings and images
(see :ref:`usage/image` and :ref:`usage/mapping`),
used by the :class:`AFQ.api.group.GroupAFQ` object. All definitions ultimately
inherit from :class:`AFQ.definitions.utils.Definition`. For a given subject and
session within the API, a definition is used to create a given mask, map, etc.
`Definition`-inherited classes must have two methods: `__init__` and `find_path`.
The requirements of each of these methods are described below:

- `Definition`-inherited classes have an `__init__` method which the users uses to configure
  the the definition for a given instance of the :class:`AFQ.api.group.GroupAFQ` object.
  These `__init__` methods must be thoroughly
  documented as they are what the user interacts with. The class must have attributes
  of same name as the `__init__` args. This is important for reading/writing
  `Definition`-inherited classes as arguments to config files.

- The api calls `find_path` during the :class:`AFQ.api.group.GroupAFQ` object initialization to
  let the definition find relevant files for the given subject and session. All `find_path`
  methods have the same input: `bids_layout`, `from_path`, `subject`, `session`. See
  :func:`AFQ.definitions.image.ImageFile.find_path` for an example.
  If your definition does not need to run anything during :class:`AFQ.api.group.GroupAFQ`
  object initialization, simply override this method with a method that only
  passes (see :func:`AFQ.definitions.image.FullImage.find_path` source for an example.)

Furthermore, mapping and image definitions each have their own required methods.
Here is the mapping required method (`get_for_subses`):

- The :class:`AFQ.api.group.GroupAFQ` object calls `get_for_subses` to get the map
  during workflow construction or execution. `get_for_subses` should return a "mapping".
  The "mapping" must have `transform` and `transform_inverse` functions which each accept
  two arguments: (1) an ndarray called `data` and (2) `**kwargs`. `get_for_subses`
  takes as input self, subses_dict, reg_template, and reg_subject.

Image definitions require `get_name`, `get_image_getter`, `get_image_direct`:

- `get_name` has no inputs and returns a name which should uniquely identify
  this image.
- `get_image_getter` returns a method which can be called as task in the task
  workflow specified by its one input, `task_name`. This method can have any valid
  inputs for its task module and ouputs an image.
- `get_image_direct` returns the image. It is similar to
  `get_image_getter`, but is called directly instead of being a task. Thus there
  are stricter restrictions on its inputs (see :func:`AFQ.definitions.image.B0Image.get_image_getter`
  for an example). Some images cannot be generated using this method because they
  rely on later derivatives. In this case, raise a value error.
