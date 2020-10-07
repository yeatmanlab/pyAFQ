def _no_tqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable to silence `tqdm`
    """
    return iterable


def _no_progressbar(progress, total_length):
    """
    no-op calls to update the `fetcher` progress bar
    """
    return


def reset_progressbars(gallery_conf, fname):
    """
    monkey patch to disable various progress bar output for examples. the
    progress bar updates pollutes sphinx gallery output. using this monkey
    patch from the spinx-build will leave the progress bars in place for other
    uses.
    """

    # disable tqdm
    import AFQ.data as afd
    import AFQ._fixes as fixes
    import AFQ.segmentation as seg
    import AFQ.viz.utils as utils

    afd.tqdm = _no_tqdm
    fixes.tqdm = _no_tqdm
    seg.tqdm = _no_tqdm
    utils.tqdm = _no_tqdm

    # disable update_progressbar
    from dipy.data import fetcher

    fetcher.update_progressbar = _no_progressbar
