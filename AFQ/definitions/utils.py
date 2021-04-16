__all__ = ["Definition", "find_file"]


class Definition(object):
    '''
    All Definitions should inherit this.
    For a given subject and session within the API, the definition is used
    to create a given mask, map, etc.
    Definitions have an init function which the users uses to specify
    how they want the definition to behave.
    The find_path and get_for_subses functions are called by the AFQ API.
    The api calls find_path to let the definition find relevant files
    for the given subject and session. The api calls get_for_subses to get the
    mask, map, etc.
    '''

    def __init__(self):
        raise NotImplementedError("Please implement an __init__ method")

    def find_path(self, bids_layout, from_path, subject, session):
        raise NotImplementedError("Please implement a find_path method")

    def str_for_toml(self):
        """
        Uses __init__ in str_for_toml to make string that will instantiate
        itself. Assumes object will have attributes of same name as
        __init__ args. This is important for reading/writing definitions
        as arguments to config files.
        """
        return type(self).__name__\
            + "("\
            + _arglist_to_string(
                self.__init__.__code__.co_varnames,
                get_attr=self)\
            + ')'


def _arglist_to_string(args, get_attr=None):
    '''
    Helper function
    Takes a list of arguments and unfolds them into a string.
    If get_attr is not None, it will be used to get the attribute
    corresponding to each argument instead.
    '''
    to_string = ""
    for arg in args:
        if arg == "self":
            continue
        if get_attr is not None:
            arg = getattr(get_attr, arg)
        if isinstance(arg, Definition):
            arg = arg.str_for_toml()
        elif isinstance(arg, str):
            arg = f"\"{arg}\""
        elif isinstance(arg, list):
            arg = "[" + _arglist_to_string(arg) + "]"
        to_string = to_string + str(arg) + ', '
    if to_string[-2:] == ', ':
        to_string = to_string[:-2]
    return to_string


def find_file(bids_layout, path, filters, suffix, session, subject,
              extension=".nii.gz"):
    """
    Helper function
    Generic calls to get_nearest to find a file
    """
    # First, try to match the session.
    nearest = bids_layout.get_nearest(
        path,
        **filters,
        extension=extension,
        suffix=suffix,
        session=session,
        subject=subject,
        full_search=True,
        strict=False,
    )

    if nearest is None:
        # If that fails, loosen session restriction
        nearest = bids_layout.get_nearest(
            path,
            **filters,
            extension=extension,
            suffix=suffix,
            subject=subject,
            full_search=True,
            strict=False,
        )

    path_subject = bids_layout.parse_file_entities(path).get(
        "subject", None
    )
    file_subject = bids_layout.parse_file_entities(nearest).get(
        "subject", None
    )
    if path_subject != file_subject:
        raise ValueError(
            f"Expected subject IDs to match for the retrieved mask file "
            f"and the supplied `from_path` file. Got sub-{file_subject} "
            f"from mask file {nearest} and sub-{path_subject} "
            f"from `from_path` file {path}."
        )

    return nearest
