import tempfile
import os.path as op
import numpy as np
import IPython.display as display

import nibabel as nib
from dipy.viz.colormap import line_colors
from dipy.viz import window, actor
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts

from palettable.tableau import Tableau_20

import AFQ.utils.volume as auv
import AFQ.registration as reg



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

    return ren


def visualize_bundles(trk, affine_or_mapping=None, bundle=None, ren=None, color=None,
                      inline=True, interact=False):
    """
    Visualize bundles in 3D using VTK
    """


    if isinstance(trk, str):
        trk = nib.streamlines.load(trk)
        tg = trk.tractogram
    else:
        # Assume these are streamlines (as list or Streamlines object):
        tg = nib.streamlines.Tractogram(trk)

    if affine_or_mapping is not None:
        tg = tg.apply_affine(np.linalg.inv(affine_or_mapping))

    streamlines = tg.streamlines

    if ren is None:
        ren = window.Renderer()

    # There are no bundles in here:
    if list(tg.data_per_streamline.keys()) == []:
        streamlines = list(streamlines)
        sl_actor = actor.line(streamlines, line_colors(streamlines))
        ren.add(sl_actor)

    if bundle is None:
        for b in np.unique(tg.data_per_streamline['bundle']):
            idx = np.where(tg.data_per_streamline['bundle'] == b)[0]
            this_sl = list(streamlines[idx])
            if color is not None:
                sl_actor = actor.line(this_sl, color)
            else:
                sl_actor = actor.line(this_sl,
                                      Tableau_20.colors[np.mod(20, int(b))])
            ren.add(sl_actor)
    else:
        idx = np.where(tg.data_per_streamline['bundle'] == bundle)[0]
        this_sl = list(streamlines[idx])
        if color is not None:
            sl_actor = actor.line(this_sl, color)
        else:
            sl_actor = actor.line(this_sl,
                                Tableau_20.colors[np.mod(20, int(bundle))])
        ren.add(sl_actor)


    return _inline_interact(ren, inline, interact)


def visualize_roi(roi, affine_or_mapping=None, static_img=None,
                  roi_affine=None, static_affine=None, reg_template=None,
                  ren=None, color=None, inline=True, interact=False):
    """
    Render a region of interest into a VTK viz as a volume
    """
    if not isinstance(roi, np.ndarray):
        if isinstance(roi, str):
            roi = nib.load(roi).get_data()
        else:
            roi = roi.get_data()

    if affine_or_mapping is not None:
        if isinstance(affine_or_mapping, np.ndarray):
            # This is an affine:
            if (static_img is None or roi_affine is None or
                  static_affine is None):
                raise ValueError("If using an affine to transform an ROI, "
                                 "need to also specify all of the following",
                                  "inputs: `static_img`, `roi_affine`, ",
                                  "`static_affine`")
            roi = reg.resample(roi, static_img, roi_affine, static_affine)
        else:
            # Assume it is  a mapping:
            if (isinstance(affine_or_mapping, str) or
                  isinstance(affine_or_mapping, nib.Nifti1Image)):
                if reg_template is None or static_img is None:
                    raise ValueError(
                        "If using a mapping to transform an ROI, need to ",
                        "also specify all of the following inputs: ",
                        "`reg_template`, `static_img`")
                affine_or_mapping = reg.read_mapping(affine_or_mapping,
                                                     static_img,
                                                     reg_template)

            roi = auv.patch_up_roi(affine_or_mapping.transform_inverse(
                                        roi,
                                        interpolation='nearest')).astype(bool)

    if ren is None:
        ren = window.ren()

    roi_actor = actor.contour_from_roi(roi, color=color)
    ren.add(roi_actor)

    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        window.record(ren, out_path=fname)
        display.display_png(display.Image(fname))

    return _inline_interact(ren, inline, interact)


def visualize_volume(volume, x=None, y=None, z=None, ren=None, inline=True,
                     interact=False):
    """
    Visualize a volume
    """
    if x is None:
        x = volume.shape[0] // 2
    if y is None:
        y = volume.shape[1] // 2
    if z is None:
        z = volume.shape[2] // 2

    vol_actor1 = actor.slicer(volume)
    vol_actor1.display(x=x)
    vol_actor2 = vol_actor1.copy()
    vol_actor2.display(y=y)
    vol_actor3 = vol_actor1.copy()
    vol_actor3.display(z=z)

    if ren is None:
        ren = window.ren()

    ren.add(vol_actor1)
    ren.add(vol_actor2)
    ren.add(vol_actor3)

    return _inline_interact(ren, inline, interact)