import multiprocessing
from joblib import Parallel, delayed


def parfor(func, in_list, out_shape=None, n_jobs=-1, func_args=[],
           func_kwargs={}):
    """
    Parallel for loop for numpy arrays
    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        func(arr, idx, *args, *kwargs) where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of `func` needs to be one
        item (e.g. float, int) per input item.
    in_list : list
       All legitimate inputs to the function to operate over.
    n_jobs : integer, optional
        The number of jobs to perform in parallel. -1 to use all cpus
        Default: 1
    args : list, optional
        Positional arguments to `func`
    kwargs : list, optional
        Keyword arguments to `func`
    Returns
    -------
    ndarray of identical shape to `arr`
    Examples
    --------
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        n_jobs=n_jobs-1

    p = Parallel(n_jobs=n_jobs, backend="threading")
    d = delayed(func)
    d_l = []
    for in_element in in_list:
        d_l.append(d(in_element, *func_args, **func_kwargs))
    results = p(d_l)

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
