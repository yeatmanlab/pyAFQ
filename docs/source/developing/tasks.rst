Adding Tasks to the pyAFQ Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pyAFQ provides methods for performing tractography, registration, bundle
recognition, visualization, tract profiling, and other elements of tractometry.
Additionally, pyAFQ provides a simple UI (which can be accessed as an API or
CLI) for running part or all of the tractometry pipeline on a given dataset.

This UI uses `Pimms <http://bids.neuroimaging.io/>`_ (a Python immutable
data structures library) as its workflow engine. Individual tasks in this workflow
are specified in the AFQ/tasks folder. The tasks are split up into 5 files in the
AFQ/tasks folder:
`data.py`, `mapping.py`, `tractography.py`, `segmentation.py`, and `viz.py`. If you want
to add new functionality to the pyAFQ workflow, first decide which file to
put it in. Try to keep similar functionality in the same file. All tasks are
decorated with `@pimms.calc("output_name")`, where `output_name` is used as
input to other tasks and can be accessed by the user through the
:class:`AFQ.api.AFQ` object.

When a user instantiates an :class:`AFQ.api.AFQ` object, a Pimms plan is
created using a selection of tasks based on the configuration provided by
the user. Each task file has a `get_{task filename}_plan` method which is
called by the :class:`AFQ.api.AFQ` class (for example, in AFQ.tasks.data
there is: :func:`AFQ.tasks.data.get_data_plan`). This method uses the user's
configuration choices (which it takes as input) to construct a Pimms plan
from the tasks in the file. In most cases, new tasks can simply be added to
the list of tasks being passed to `with_name`. These tasks are always added
to the plan; however they will only be called if the user needs them. In a few
cases, additional logic is required, like when two different tasks compute the
same output, or if the task being provided by a `Definition`-inherited object
(see the source of :func:`AFQ.tasks.mapping.get_mapping_plan` for an example
of both).

The outputs of each task can be accessed by the user through the
AFQ object, either as an attribute or using a `export_` method,
such as `export_b0`. In each of the 5 tasks files, there is a variable called
`outputs <https://github.com/yeatmanlab/pyAFQ/blob/7204682b22af1c8c89766dacfd25ec01bcce9442/AFQ/tasks/data.py#L26>`_.
Add your task output(s) to that variable for them to be attached to
the AFQ class. Note that when an output is attached to the AFQ class either as
an attribute or method, if that output name ends in '_file',
the '_file' is removed from the name automatically. 

Task Decorators
~~~~~~~~~~~~~~~

When adding a new task, check AFQ.tasks.decorators to see if any of those
decorators would be useful. Here are descriptions of some useful decorators:

- :func:`AFQ.tasks.decorators.as_file`: in pyAFQ, we often save derivatives to
  the disk so they can be reused in later runs. This decorator implements an if
  statement: if the file already exists, return that, otherwise generate the file
  with a BIDS-compliant filename using the output of the task. To use this
  decorator, your task must have as input subses_dict and return two objects:
  (1) either a Nifti1Image, StatefulTractogram, or pandas dataframe, and (2)
  a dictionary containing metadata. Provide the decorator with the suffix you
  want the file to have and whether or not tracking or segmentation information
  should be included in the BIDS-compliant filename.

- :func:`AFQ.tasks.decorators.as_model`: this decorator is useful for
  implementing ODF models. It adds timing information
  to the metadata dictionary and converts model data into a Nifti1Image, both
  of which can then be passed to :func:`AFQ.tasks.decorators.as_file`. To use
  this decorator, your task must have as input dwi_affine and return two objects:
  (1) model data as an ndarray and (2) a dictionary containing metadata.

- :func:`AFQ.tasks.decorators.as_dt_deriv`: this decorator is useful for
  implementing both DTI and DKI derivatives. You provide it with which model
  you are using, either "DTI" or "DKI". It creates a metadata dictionary using
  the location of the model params file
  and converts the derivative data into a Nifti1Image, both
  of which can then be passed to :func:`AFQ.tasks.decorators.as_file`. To use
  this decorator, your task must have as input dwi_affine and either
  dki_params_file or dti_params_file. Your task must return the derivative data
  as an ndarray.

- :func:`AFQ.tasks.decorators.as_img`: this decorator simply converts the ndarray
  output of a task into a Nifti1Image, typically to be passed to
  :func:`AFQ.tasks.decorators.as_file`. To use this decorator, your task must
  have as input with affine in its name. Your decorator must return two objects:
  (1) data as an ndarray and (2) a dictionary containing metadata.

Checklist for Adding Tasks to the pyAFQ Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is what you must do each time you add a task to the workflow: 

#. Add your task to the relevant tasks file with the pimms.calc decorator.

#. Check AFQ.tasks.decorators for useful decorators that may simplify the code.

#. Add your task to the `get_{task filename}_plan` method.

#. Add outputs of the task to the `outputs` variable in its task file.

In most cases, you should only have to edit the tasks file which you are adding to.
The API (:class:`AFQ.api.AFQ`) automatically reads these files to construct its
workflow.
