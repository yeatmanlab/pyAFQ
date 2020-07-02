import tempfile
import os
import os.path as op
import enum
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import plotly
import plotly.graph_objs as go
import imageio as io
from palettable.tableau import Tableau_20

import nibabel as nib
import dipy.tracking.streamlinespeed as dps

import AFQ.utils.volume as auv
import AFQ.registration as reg

tableau_20_rgb = np.array(Tableau_20.colors) / 255 - 0.0001

COLOR_DICT = OrderedDict({"ATR_L": tableau_20_rgb[0],
                          "ATR_R": tableau_20_rgb[1],
                          "CST_L": tableau_20_rgb[2],
                          "CST_R": tableau_20_rgb[3],
                          "CGC_L": tableau_20_rgb[4],
                          "CGC_R": tableau_20_rgb[5],
                          "HCC_L": tableau_20_rgb[6],
                          "HCC_R": tableau_20_rgb[7],
                          "FP": tableau_20_rgb[8],
                          "FA": tableau_20_rgb[9],
                          "IFO_L": tableau_20_rgb[10],
                          "IFO_R": tableau_20_rgb[11],
                          "ILF_L": tableau_20_rgb[12],
                          "ILF_R": tableau_20_rgb[13],
                          "SLF_L": tableau_20_rgb[14],
                          "SLF_R": tableau_20_rgb[15],
                          "UNC_L": tableau_20_rgb[16],
                          "UNC_R": tableau_20_rgb[17],
                          "ARC_L": tableau_20_rgb[18],
                          "ARC_R": tableau_20_rgb[19]})

POSITIONS = OrderedDict({"ATR_L": (1, 0), "ATR_R": (1, 4),
                         "CST_L": (1, 1), "CST_R": (1, 3),
                         "CGC_L": (3, 1), "CGC_R": (3, 3),
                         "HCC_L": (4, 1), "HCC_R": (4, 3),
                         "FP": (4, 2), "FA": (0, 2),
                         "IFO_L": (4, 0), "IFO_R": (4, 4),
                         "ILF_L": (3, 0), "ILF_R": (3, 4),
                         "SLF_L": (2, 1), "SLF_R": (2, 3),
                         "ARC_L": (2, 0), "ARC_R": (2, 4),
                         "UNC_L": (0, 1), "UNC_R": (0, 3)})


def _inline_interact(figure, show, show_inline):
    """
    Helper function to reuse across viz functions
    """
    if show:
        plotly.offline.plot(figure)

    if show_inline:
        plotly.offline.iplot(figure)

    return figure


def _color_arr2str(color_arr, opacity=1.0):
    return f"rgba({color_arr[0]}, {color_arr[1]}, {color_arr[2]}, {opacity})"


def _draw_streamlines(figure, sls, color, name, n_points=100):
    x_pts = []
    y_pts = []
    z_pts = []

    for sl in sls:
        # resample streamline to n_points
        if sl.shape[0] > n_points:
            sl = dps.set_number_of_points(sl, n_points)

        # add sl to lines
        x_pts.extend(sl[:, 0])
        x_pts.append(None)  # don't draw between streamlines
        y_pts.extend(sl[:, 1])
        y_pts.append(None)
        z_pts.extend(sl[:, 2])
        z_pts.append(None)

    figure.add_trace(
        go.Scatter3d(
            x=x_pts,
            y=y_pts,
            z=z_pts,
            mode='lines',
            name=name,
            line=dict(
                width=1,
                color=_color_arr2str(color),
            )
        )
    )


def visualize_bundles(trk, affine=None, bundle_dict=None, bundle=None,
                      colors=None, figure=None, background=(1, 1, 1),
                      resample=100, show=False, show_inline=False):
    """
    Visualize bundles in 3D using VTK

    Parameters
    ----------
    trk : str, list, or Streamlines
        The streamline information

    affine : ndarray, optional
       An affine transformation to apply to the streamlines before
       visualization. Default: no transform.

    bundle_dict : dict, optional
        Keys are names of bundles and values are dicts that should include
        a key `'uid'` with values as integers for selection from the trk
        metadata. Default: bundles are either not identified, or identified
        only as unique integers in the metadata.

    bundle : str or int, optional
        The name of a bundle to select from among the keys in `bundle_dict`
        or an integer for selection from the trk metadata.

    colors : dict or list
        If this is a dict, keys are bundle names and values are RGB tuples.
        If this is a list, each item is an RGB tuple. Defaults to a list
        with Tableau 20 RGB values

    background : tuple, optional
        RGB values for the background. Default: (1, 1, 1), which is white
        background.

    scene : fury Scene object, optional
        If provided, the visualization will be added to this Scene. Default:
        Initialize a new Scene.

    show : bool
        Whether to provide an interactive html file.
        Default: False

    show_inline : bool
        Whether to embed the visualization inline in a notebook. Only works
        in the notebook context. Default: False.

    Returns
    -------
    Fury Scene object
    """
    if isinstance(trk, str):
        trk = nib.streamlines.load(trk)
        tg = trk.tractogram
    else:
        # Assume these are streamlines (as list or Streamlines object):
        tg = nib.streamlines.Tractogram(trk)

    if affine is not None:
        tg = tg.apply_affine(np.linalg.inv(affine))

    streamlines = tg.streamlines

    if figure is None:
        figure = go.Figure()

    figure.update_layout(plot_bgcolor=_color_arr2str(background))

    if colors is None:
        # Use the color dict provided
        colors = COLOR_DICT

    def _bundle_selector(bundle_dict, colors, b):
        """Helper function """
        b_name = str(b)
        if bundle_dict is None:
            # We'll choose a color from a rotating list:
            if isinstance(colors, list):
                color = colors[np.mod(len(colors), int(b))]
            else:
                color_list = colors.values()
                color = color_list[np.mod(len(colors), int(b))]
        else:
            # We have a mapping from UIDs to bundle names:
            for b_name_iter, b_iter in bundle_dict.items():
                if b_iter['uid'] == b:
                    b_name = b_name_iter
                    break
            color = colors[b_name]
        return color, b_name

    if list(tg.data_per_streamline.keys()) == []:
        # There are no bundles in here:
        streamlines = list(streamlines)
        # Visualize all the streamlines with directionally assigned RGB:
        _draw_streamlines(figure, streamlines, [
                          0.5, 0.5, 0.5], "all_bundles", n_points=resample)

    else:
        # There are bundles:
        if bundle is None:
            # No selection: visualize all of them:

            for b in np.unique(tg.data_per_streamline['bundle']):
                idx = np.where(tg.data_per_streamline['bundle'] == b)[0]
                these_sls = list(streamlines[idx])
                color, b_name = _bundle_selector(bundle_dict, colors, b)
                _draw_streamlines(figure, these_sls, color,
                                  b_name, n_points=resample)

        else:
            # Select just one to visualize:
            if isinstance(bundle, str):
                # We need to find the UID:
                uid = bundle_dict[bundle]['uid']
            else:
                # It's already a UID:
                uid = bundle

            idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
            these_sls = list(streamlines[idx])
            color, b_name = _bundle_selector(bundle_dict, colors, uid)
            _draw_streamlines(figure, these_sls, color,
                              b_name, n_points=resample)

    return _inline_interact(figure, show, show_inline)


def stop_orca():
    plotly.io.orca.shutdown_server()


def create_gif(figure, file_name, n_frames=60, zoom=2.5, z_offset=0.5, auto_stop_orca=True):
    tdir = tempfile.gettempdir()

    for i in range(n_frames):
        theta = (i * 6.28) / n_frames
        camera = dict(
            eye=dict(x=np.cos(theta) * zoom,
                     y=np.sin(theta) * zoom, z=z_offset)
        )
        figure.update_layout(scene_camera=camera)
        figure.write_image(tdir + f"/tgif{i}.png")

    if auto_stop_orca:
        stop_orca()

    angles = []
    for i in range(n_frames):
        if i < 10:
            angle_fname = f"tgif{i}.png"
        elif i < 100:
            angle_fname = f"tgif{i}.png"
        else:
            angle_fname = f"tgif{i}.png"
        angles.append(io.imread(os.path.join(tdir, angle_fname)))

    io.mimsave(file_name, angles)


def visualize_gif_inline(fname, use_s3fs=False):
    if use_s3fs:
        import s3fs
        fs = s3fs.S3FileSystem()
        tdir = tempfile.gettempdir()
        fname_remote = fname
        fname = op.join(tdir, "fig.gif")
        fs.get(fname_remote, fname)

    display.display(display.Image(fname))


def _draw_roi(figure, roi, color, opacity):
    roi = np.where(roi == 1)
    figure.add_trace(
        go.Scatter3d(
            x=roi[0],
            y=roi[1],
            z=roi[2],
            marker=dict(color=_color_arr2str(color, opacity=opacity)),
            line=dict(color=f"rgba(0,0,0,0)")
        )
    )


def visualize_roi(roi, affine_or_mapping=None, static_img=None,
                  roi_affine=None, static_affine=None, reg_template=None,
                  figure=None, color=np.array([0.9999, 0, 0]), opacity=1.0,
                  show=False, show_inline=False):
    """
    Render a region of interest into a VTK viz as a volume
    """
    if not isinstance(roi, np.ndarray):
        if isinstance(roi, str):
            roi = nib.load(roi).get_fdata()
        else:
            roi = roi.get_fdata()

    if affine_or_mapping is not None:
        if isinstance(affine_or_mapping, np.ndarray):
            # This is an affine:
            if (static_img is None or roi_affine is None
                    or static_affine is None):
                raise ValueError("If using an affine to transform an ROI, "
                                 "need to also specify all of the following",
                                 "inputs: `static_img`, `roi_affine`, ",
                                 "`static_affine`")
            roi = reg.resample(roi, static_img, roi_affine, static_affine)
        else:
            # Assume it is  a mapping:
            if (isinstance(affine_or_mapping, str)
                    or isinstance(affine_or_mapping, nib.Nifti1Image)):
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

    if figure is None:
        figure = go.Figure()

    figure.update_layout(plot_bgcolor=f"rgba(0,0,0,0)")

    _draw_roi(figure, roi, color, opacity)

    return _inline_interact(figure, show, show_inline)


class Axes(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


def _draw_slice(figure, axis, volume, opacity=0.3, step=None, n_steps=0):
    if step is None:
        height = volume.shape[axis] // 2
        visible = True
    else:
        height = (volume.shape[axis] * step) // n_steps
        visible = False

    if axis == Axes.X:
        X, Y, Z = np.mgrid[height:height + 1,
                           :volume.shape[1], :volume.shape[2]]
        values = 1 - volume[height, :, :].flatten()
    elif axis == Axes.Y:
        X, Y, Z = np.mgrid[:volume.shape[0],
                           height:height + 1, :volume.shape[2]]
        values = 1 - volume[:, height, :].flatten()
    elif axis == Axes.Z:
        X, Y, Z = np.mgrid[:volume.shape[0],
                           :volume.shape[1], height:height + 1]
        values = 1 - volume[:, :, height].flatten()

    figure.add_trace(
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values,
            colorscale='greys',
            surface_count=1,
            showscale=False,
            opacity=opacity,
            visible=visible
        )
    )


def _name_from_enum(axis):
    if axis == Axes.X:
        return "Coronal"
    elif axis == Axes.Y:
        return "Sagittal"
    elif axis == Axes.Z:
        return "Axial"


def _draw_slices(figure, axis, volume, opacity=0.3, sliders=[], n_steps=0, y_loc=0):
    if n_steps == 0:
        _draw_slice(figure, axis, volume, opacity=opacity)
    else:
        active = n_steps // 2
        name = _name_from_enum(axis) + " Plane"
        steps = []
        for step in range(n_steps):
            _draw_slice(figure, axis, volume, opacity=opacity,
                        step=step, n_steps=n_steps)

            step_dict = dict(
                method="update",
                args=[{"visible": [False] * n_steps}],
                label=""
            )

            step_dict["args"][0]["visible"][step] = True
            steps.append(step_dict)

        figure.data[-active].visible = True
        sliders.append(dict(
            active=active,
            currentvalue=dict(visible=True, prefix=name, xanchor='center'),
            pad=dict(t=50),
            steps=steps,
            y=y_loc,
            x=0.2,
            lenmode='fraction',
            len=0.6
        ))  # TODO: these sliders won't become independent!


def visualize_volume(volume, figure=None, show_x=True, show_y=True, show_z=True,
                     show=False, opacity=0.3, show_inline=False, slider_definition=0):
    """
    Visualize a volume
    """
    if isinstance(volume, str):
        volume = nib.load(volume).get_fdata()

    if figure is None:
        figure = go.Figure()

    figure.update_layout(plot_bgcolor=f"rgba(0,0,0,0)")
    sliders = []

    if show_x:
        _draw_slices(figure, Axes.X, volume, sliders=sliders,
                     opacity=opacity, n_steps=slider_definition, y_loc=0)
    if show_y:
        _draw_slices(figure, Axes.Y, volume, sliders=sliders,
                     opacity=opacity, n_steps=slider_definition, y_loc=-0.3)
    if show_z:
        _draw_slices(figure, Axes.Z, volume, sliders=sliders,
                     opacity=opacity, n_steps=slider_definition, y_loc=-0.6)

    figure.update_layout(sliders=tuple(sliders))

    return _inline_interact(figure, show, show_inline)


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
