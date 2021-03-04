__all__ = ["check_definition_methods", "StrInstantiatesMixin", "find_file"]


def check_definition_methods(definition, definition_name=False):
    '''
    Helper function
    Checks if definition is a valid definition.
    If definition_name is not False, will throw an error stating the method
    not found and the mask name.
    '''
    if not hasattr(definition, 'find_path'):
        if definition_name:
            raise TypeError(
                f"find_path method not found in {definition_name}")
        else:
            return False
    elif not hasattr(definition, 'get_for_row'):
        if definition_name:
            raise TypeError(
                f"get_for_row method not found in {definition_name}")
        else:
            return False
    elif not hasattr(definition, '__init__')\
            or not hasattr(definition.__init__, '__code__'):
        if definition_name:
            raise TypeError(
                f"__init__ method not defined in {definition_name}")
        else:
            return False
    else:
        return True


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
        if check_definition_methods(arg):
            arg = arg.str_for_toml()
        elif isinstance(arg, str):
            arg = f"\"{arg}\""
        elif isinstance(arg, list):
            arg = "[" + _arglist_to_string(arg) + "]"
        to_string = to_string + str(arg) + ', '
    if to_string[-2:] == ', ':
        to_string = to_string[:-2]
    return to_string


class StrInstantiatesMixin(object):
    '''
    Helper class
    Uses __init__ in str_for_toml to make string that will instantiate itself.
    Assumes object will have attributes of same name as __init__ args.
    This is important for reading/writing definitions as arguments
    to config files.
    '''

    def str_for_toml(self):
        return type(self).__name__\
            + "("\
            + _arglist_to_string(
                self.__init__.__code__.co_varnames,
                get_attr=self)\
            + ')'


def find_file(bids_layout, path, filters, suffix, session, subject):
    """
    Helper function
    Generic calls to get_nearest to find a file
    """
    # First, try to match the session.
    nearest = bids_layout.get_nearest(
        path,
        **filters,
        extension=".nii.gz",
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
            extension=".nii.gz",
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
