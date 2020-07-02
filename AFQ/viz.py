import tempfile
import os.path as op

import IPython.display as display
import matplotlib.pyplot as plt

from AFQ.viz_libs.utils import POSITIONS, COLOR_DICT


class Viz:
    def __init__(self,
                 viz_library="fury"):
        """
        Set up visualization preferences.

        Parameters
        ----------
            viz_library : str, optional
                Should be either "fury" or "plotly".
                Default: "fury"
        """
        if viz_library == "fury":
            import AFQ.viz_libs.afq_fury
            self.visualize_bundles = AFQ.viz_libs.afq_fury.visualize_bundles
            self.visualize_roi = AFQ.viz_libs.afq_fury.visualize_roi
            self.visualize_volume = AFQ.viz_libs.afq_fury.visualize_volume
            self.create_gif = AFQ.viz_libs.afq_fury.create_gif
            self.stop_creating_gifs = AFQ.viz_libs.afq_fury.stop_creating_gifs
        elif viz_library == "plotly":
            import AFQ.viz_libs.afq_plotly
            self.visualize_bundles = AFQ.viz_libs.afq_plotly.visualize_bundles
            self.visualize_roi = AFQ.viz_libs.afq_plotly.visualize_roi
            self.visualize_volume = AFQ.viz_libs.afq_plotly.visualize_volume
            self.create_gif = AFQ.viz_libs.afq_plotly.create_gif
            self.stop_creating_gifs = AFQ.viz_libs.afq_plotly.stop_creating_gifs


def visualize_tract_profiles(tract_profiles, scalar="dti_fa", min_fa=0.0,
                             max_fa=1.0, file_name=None, positions=POSITIONS):
    """
    Visualize all tract profiles for a scalar in one plot

    Parameters
    ----------
    tract_profiles : pandas dataframe
        Pandas dataframe of tract_profiles. For example,
        tract_profiles = pd.read_csv(my_afq.get_tract_profiles()[0])

    scalar : string, optional
       Scalar to use in plots. Default: "dti_fa".

    min_fa : float, optional
        Minimum FA used for y-axis bounds. Default: 0.0

    max_fa : float, optional
        Maximum FA used for y-axis bounds. Default: 1.0

    file_name : string, optional
        File name to save figure to if not None. Default: None

    positions : dictionary, optional
        Dictionary that maps bundle names to position in plot.
        Default: POSITIONS

    Returns
    -------
        Matplotlib figure and axes.
    """

    if (file_name is not None):
        plt.ioff()

    fig, axes = plt.subplots(5, 5)

    for bundle in positions.keys():
        ax = axes[positions[bundle][0], positions[bundle][1]]
        fa = tract_profiles[
            (tract_profiles["bundle"] == bundle)
        ][scalar].values
        ax.plot(fa, 'o-', color=COLOR_DICT[bundle])
        ax.set_ylim([min_fa, max_fa])
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_yticklabels([0.2, 0.4, 0.6])
        ax.set_xticklabels([])

    fig.set_size_inches((12, 12))

    axes[0, 0].axis("off")
    axes[0, -1].axis("off")
    axes[1, 2].axis("off")
    axes[2, 2].axis("off")
    axes[3, 2].axis("off")

    if (file_name is not None):
        fig.savefig(file_name)
        plt.ion()

    return fig, axes


def visualize_gif_inline(fname, use_s3fs=False):
    """Display a gif inline, possible from s3fs """
    if use_s3fs:
        import s3fs
        fs = s3fs.S3FileSystem()
        tdir = tempfile.gettempdir()
        fname_remote = fname
        fname = op.join(tdir, "fig.gif")
        fs.get(fname_remote, fname)

    display.display(display.Image(fname))
