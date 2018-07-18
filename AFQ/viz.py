import tempfile
import os.path as op
import numpy as np
import IPython.display as display

import nibabel as nib
from dipy.viz.colormap import line_colors
from dipy.viz import window, actor

from palettable.tableau import Tableau_20

def _inline_interact(ren, inline, interact):
    """
    Helper function to reuse across viz functions
    """
    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        window.record(ren, out_path=fname)
        display.display_png(display.Image(fname))

    if interact:
        window.show(ren)

    return RuntimeError


def visualize_bundles(trk, ren=None, inline=True, interact=False):
    """
    Visualize bundles in 3D using VTK
    """
    if isinstance(trk, str):
        trk = nib.streamlines.load(trk)

    if ren is None:
        ren = window.Renderer()

    # There are no bundles in here:
    if list(trk.tractogram.data_per_streamline.keys()) == []:
        streamlines = list(trk.streamlines)
        sl_actor = actor.line(streamlines, line_colors(streamlines))
        ren.add(sl_actor)

    for b in np.unique(trk.tractogram.data_per_streamline['bundle']):
        idx = np.where(trk.tractogram.data_per_streamline['bundle'] == b)[0]
        this_sl = list(trk.streamlines[idx])
        sl_actor = actor.line(this_sl, Tableau_20.colors[np.mod(20, int(b))])
        ren.add(sl_actor)

    return _inline_interact(ren, inline, interact)


def visualize_roi(roi, ren=None, inline=True, interact=False):
    """
    Render a region of interest into a VTK viz as a volume
    """

    if ren is None:
        ren = window.ren()

    roi_actor = actor.contour_from_roi(roi)
    ren.add(roi_actor)

    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        window.record(ren, out_path=fname)
        display.display_png(display.Image(fname))

    return _inline_interact(ren, inline, interact)


def visualize_volume(volume, x=None, y=None, z=None):
    """
    Visualize a volume
    """

    vol_actor = actor.slicer(t1_data)
