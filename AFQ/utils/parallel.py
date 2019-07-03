import numpy as np
import multiprocessing
import joblib
import dask
import dask.multiprocessing
from distributed import Client
from tqdm import tqdm


def parfor(func, in_list, out_shape=None, n_jobs=-1, engine="joblib",
           backend="threading", client=None, func_args=[], func_kwargs={}):
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
        The number of jobs to perform in parallel. -1 to use all cpus.
        Default: 1
    engine : str
        {"dask", "joblib", "serial"}
        The last one is useful for debugging -- runs the code without any
        parallelization.
    backend : str
        What joblib backend or dask scheduler to use.
        One of {"threading" | "multiprocessing" | "processes" | "threads"}
    client : a distributed.Client instance.
        For the dask engine, a distributed client to use for mapping jobs.
    func_args : list, optional
        Positional arguments to `func`.
    func_kwargs : list, optional
        Keyword arguments to `func`.

    Returns
    -------
    ndarray of identical shape to `arr`

    Examples
    --------

    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        n_jobs = n_jobs - 1

    if engine == "joblib":
        p = joblib.Parallel(n_jobs=n_jobs, backend=backend)
        d = joblib.delayed(func)
        d_l = []
        for in_element in in_list:
            d_l.append(d(in_element, *func_args, **func_kwargs))
        results = p(d_l)

    elif engine == "dask":
        def partial(func, *args, **keywords):
            def newfunc(in_arg):
                return func(in_arg, *args, **keywords)
            return newfunc
        p = partial(func, *func_args, **func_kwargs)
        if client is None:
            client = Client()
        results = client.map(p, in_list)

        all_done = False
        pbar = tqdm(total=len(in_list))
        n_done = 0
        while not all_done:
            n_done_now = sum([r.done() for r in results])
            if n_done_now > n_done:
                pbar.update(n_done_now - n_done)
                n_done = n_done_now

            all_done = n_done == len(in_list)

        exceptions = {}
        for ii, rr in enumerate(results):
            if rr.status == 'error':
                exceptions[ii] = rr.exception()

    elif engine == "serial":
        results = []
        for in_element in in_list:
            results.append(func(in_element, *func_args, **func_kwargs))

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
