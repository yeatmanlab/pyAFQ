import tempfile
import os.path as op
import numpy as np
import IPython.display as display

import nibabel as nib
from dipy.viz import fvtk
from palettable.tableau import Tableau_20


def visualize_bundles(trk, ren=None, inline=True, interact=False):
    """
    Visualize bundles in 3D using fvtk


    """
    if isinstance(trk, str):
        trk = nib.streamlines.load(trk)

    if ren is None:
        ren = fvtk.ren()

    for b in np.unique(trk.tractogram.data_per_streamline['bundle']):
        idx = np.where(trk.tractogram.data_per_streamline['bundle'] == b)[0]
        this_sl = list(trk.streamlines[idx])
        sl_actor = fvtk.line(this_sl, Tableau_20.colors[np.mod(20, int(b))])
        fvtk.add(ren, sl_actor)

    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        fvtk.record(ren, out_path=fname)
        display.display_png(display.Image(fname))

    if interact:
        fvtk.show(ren)

    return ren
