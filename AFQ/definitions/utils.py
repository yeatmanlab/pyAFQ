import os.path as op
import logging

from AFQ.utils.path import drop_extension


logger = logging.getLogger('AFQ')


__all__ = ["Definition", "find_file", "name_from_path"]


class Definition(object):
    '''
    All Definitions should inherit this.
    For a given subject and session within the API, the definition is used
    to create a given image or map.
    Definitions have an init function which the users uses to specify
    how they want the definition to behave.
    The find_path function is called by the AFQ API.
    The api calls find_path to let the definition find relevant files
    for the given subject and session.
    '''

    def __init__(self):
        raise NotImplementedError("Please implement an __init__ method")

    def find_path(self, bids_layout, from_path,
                  subject, session, required=True):
        pass

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
            arg = f"[{_arglist_to_string(arg)}]"
        to_string = to_string + str(arg) + ', '
    if to_string[-2:] == ', ':
        to_string = to_string[:-2]
    return to_string


def name_from_path(path):
    file_name = op.basename(path)  # get file name
    file_name = drop_extension(file_name)  # remove extension
    if "-" in file_name:
        file_name = file_name.split("-")[-1]  # get suffix if exists
    return file_name


def _ff_helper(required, err_msg):
    if required:
        raise ValueError(err_msg)
    else:
        logger.warning(err_msg)
        return None


def find_file(bids_layout, path, filters, suffix, session, subject,
              extension=".nii.gz", required=True):
    """
    Helper function
    Generic calls to get_nearest to find a file
    """
    if "extension" not in filters:
        filters["extension"] = extension
    if "suffix" not in filters:
        filters["suffix"] = suffix

    # First, try to match the session.
    nearest = bids_layout.get_nearest(
        path,
        **filters,
        session=session,
        subject=subject,
        full_search=True,
        strict=False,
    )

    # If that fails, loosen session restriction
    # in order to find scans that are not session specific
    if nearest is None:
        nearest = bids_layout.get_nearest(
            path,
            **filters,
            subject=subject,
            full_search=True,
            strict=False,
        )

    # Nothing is found
    if nearest is None:
        return _ff_helper(required, (
            "No file found with these parameters:\n"
            f"suffix: {suffix},\n"
            f"session (searched with and without): {session},\n"
            f"subject: {subject},\n"
            f"filters: {filters},\n"
            f"near path: {path},\n"))

    path_subject = bids_layout.parse_file_entities(path).get(
        "subject", None
    )
    file_subject = bids_layout.parse_file_entities(nearest).get(
        "subject", None
    )
    path_session = bids_layout.parse_file_entities(path).get(
        "session", None
    )
    file_session = bids_layout.parse_file_entities(nearest).get(
        "session", None
    )

    # found file is wrong subject
    if path_subject != file_subject:
        return _ff_helper(required, (
            f"Expected subject IDs to match for the retrieved image file "
            f"and the supplied `from_path` file. Got sub-{file_subject} "
            f"from image file {nearest} and sub-{path_subject} "
            f"from `from_path` file {path}."))

    # found file is wrong session
    if (file_session is not None) and (path_session != file_session):
        return _ff_helper(required, (
            f"Expected session IDs to match for the retrieved image file "
            f"and the supplied `from_path` file. Got ses-{file_session} "
            f"from image file {nearest} and ses-{path_session} "
            f"from `from_path` file {path}."))

    return nearest
