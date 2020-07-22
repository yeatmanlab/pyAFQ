import tempfile
import os
import os.path as op
import enum
import logging

import numpy as np

import dipy.tracking.streamlinespeed as dps

import AFQ.viz.utils as vut

try:
    import plotly
    import plotly.graph_objs as go
except ImportError:
    raise ImportError(vut.viz_import_msg_error("plotly"))

viz_logger = logging.getLogger("AFQ.viz")


def _inline_interact(figure, show, show_inline):
    """
    Helper function to reuse across viz functions
    """
    if show:
        viz_logger.info("Creating interactive figure in HTML file...")
        plotly.offline.plot(figure)

    if show_inline:
        viz_logger.info("Creating interactive figure inline...")
        plotly.offline.init_notebook_mode()
        plotly.offline.iplot(figure)

    return figure


def _to_color_range(num):
    if num < 0:
        num = 0
    if num >= 0.999:
        num = 0.999
    return num


def _color_arr2str(color_arr, opacity=1.0):
    return (
        f"rgba({_to_color_range(color_arr[0])}, "
        f"{_to_color_range(color_arr[1])}, "
        f"{_to_color_range(color_arr[2])}, "
        f"{_to_color_range(opacity)})"
    )


def set_layout(figure, color=None):
    if color is None:
        color = f"rgba(0,0,0,0)"

    figure.update_layout(
        plot_bgcolor=color,
        scene=dict(
            xaxis=dict(
                showbackground=False, showticklabels=False, title=''),
            yaxis=dict(
                showbackground=False, showticklabels=False, title=''),
            zaxis=dict(
                showbackground=False, showticklabels=False, title='')
        )
    )


def _draw_streamlines(figure, sls, color, name, cbv=None):
    color = np.asarray(color)
    n_points = 100

    bundle_shape = ((n_points+1)*sls._offsets.shape[0])
    # dtype object so None can be stored
    x_pts = np.zeros(bundle_shape)
    y_pts = np.zeros(bundle_shape)
    z_pts = np.zeros(bundle_shape)

    if cbv is not None:
        customdata = np.zeros(bundle_shape)
        line_color = np.zeros((bundle_shape, 3))
        color_constant = (color / color.max()) * (1.4 / cbv.max())
    else:
        customdata = None
        line_color = _color_arr2str(color)

    for cumul_offset, curr_offset in enumerate(sls._offsets):
        sl = sls._data[curr_offset:curr_offset+sls._lengths[cumul_offset]]
        sl = dps.set_number_of_points(sl, n_points)

        # add sl to lines
        total_offset = (n_points+1)*cumul_offset
        x_pts[total_offset:total_offset+n_points] = sl[:, 0]
        x_pts[total_offset+n_points] = np.nan  # don't draw between streamlines
        y_pts[total_offset:total_offset+n_points] = sl[:, 1]
        y_pts[total_offset+n_points] = np.nan
        z_pts[total_offset:total_offset+n_points] = sl[:, 2]
        z_pts[total_offset+n_points] = np.nan

        if cbv is not None:
            brightness = cbv[
                sl[:, 0].astype(int),
                sl[:, 1].astype(int),
                sl[:, 2].astype(int)
            ]

            line_color[total_offset:total_offset+n_points, :] = \
                np.outer(brightness, color_constant)
            customdata[total_offset:total_offset+n_points] = brightness

            line_color[total_offset+n_points, :] = [0, 0, 0]
            customdata[total_offset+n_points] = 0

    figure.add_trace(
        go.Scatter3d(
            x=x_pts,
            y=y_pts,
            z=z_pts,
            mode='lines',
            name=name,
            line=dict(
                width=8,
                color=line_color,
            ),
            hovertext=customdata,
            hoverinfo='all'
        )
    )


def visualize_bundles(sft, affine=None, bundle_dict=None, bundle=None,
                      colors=None, color_by_volume=None, figure=None,
                      background=(1, 1, 1), interact=False, inline=False):
    """
    Visualize bundles in 3D

    Parameters
    ----------
    sft : Stateful Tractogram, str
        A Stateful Tractogram containing streamline information
        or a path to a trk file

    affine : ndarray, optional
       An affine transformation to apply to the streamlines before
       visualization. Default: no transform.

    bundle_dict : dict, optional
        Keys are names of bundles and values are dicts that should include
        a key `'uid'` with values as integers for selection from the sft
        metadata. Default: bundles are either not identified, or identified
        only as unique integers in the metadata.

    bundle : str or int, optional
        The name of a bundle to select from among the keys in `bundle_dict`
        or an integer for selection from the sft metadata.

    colors : dict or list
        If this is a dict, keys are bundle names and values are RGB tuples.
        If this is a list, each item is an RGB tuple. Defaults to a list
        with Tableau 20 RGB values

    background : tuple, optional
        RGB values for the background. Default: (1, 1, 1), which is white
        background.

    figure : Plotly Figure object, optional
        If provided, the visualization will be added to this Figure. Default:
        Initialize a new Figure.

    interact : bool
        Whether to open the visualization in an interactive window.
        Default: False

    inline : bool
        Whether to embed the interactivevisualization inline in a notebook.
        Only works in the notebook context. Default: False.

    Returns
    -------
    Plotly Figure object
    """

    if color_by_volume is not None:
        color_by_volume = vut.load_volume(color_by_volume)

    if figure is None:
        figure = go.Figure()

    set_layout(figure, color=_color_arr2str(background))

    for (sls, color, name) in \
            vut.tract_generator(sft, affine, bundle, bundle_dict, colors):
        _draw_streamlines(
            figure,
            sls,
            color,
            name,
            cbv=color_by_volume)

    return _inline_interact(figure, interact, inline)


def create_gif(figure,
               file_name,
               n_frames=60,
               zoom=2.5,
               z_offset=0.5,
               size=(600, 600)):
    """
    Convert a Plotly Figure object into a gif

    Parameters
    ----------
    figure: Plotly Figure object
        Figure to be converted to a gif

    file_name: str
        File to save gif to.

    n_frames: int, optional
        Number of frames in gif.
        Will be evenly distributed throughout the rotation.
        Default: 60

    zoom: float, optional
        How much to magnify the figure in the fig.
        Default: 2.5

    size: tuple, optional
        Size of the gif.
        Default: (600, 600)
    """
    tdir = tempfile.gettempdir()

    for i in range(n_frames):
        theta = (i * 6.28) / n_frames
        camera = dict(
            eye=dict(x=np.cos(theta) * zoom,
                     y=np.sin(theta) * zoom, z=z_offset)
        )
        figure.update_layout(scene_camera=camera)
        figure.write_image(tdir + f"/tgif{i}.png")

    vut.gif_from_pngs(tdir, file_name, n_frames,
                      png_fname="tgif", add_zeros=False)


def _draw_roi(figure, roi, name, color, opacity):
    roi = np.where(roi == 1)
    figure.add_trace(
        go.Scatter3d(
            x=roi[0] + 1,
            y=roi[1] + 1,
            z=roi[2] + 1,
            name=name,
            marker=dict(color=_color_arr2str(color, opacity=opacity)),
            line=dict(color=f"rgba(0,0,0,0)")
        )
    )


def visualize_roi(roi, affine_or_mapping=None, static_img=None,
                  roi_affine=None, static_affine=None, reg_template=None,
                  name='ROI', figure=None, color=np.array([0.9999, 0, 0]),
                  opacity=1.0, interact=False, inline=False):
    """
    Render a region of interest into a volume

    Parameters
    ----------
    roi : str or Nifti1Image
        The ROI information

    affine_or_mapping : ndarray, Nifti1Image, or str, optional
       An affine transformation or mapping to apply to the ROIs before
       visualization. Default: no transform.

    static_img: str or Nifti1Image, optional
        Template to resample roi to.
        Default: None

    roi_affine: ndarray, optional
        Default: None

    static_affine: ndarray, optional
        Default: None

    reg_template: str or Nifti1Image, optional
        Template to use for registration.
        Default: None

    name: str, optional
        Name of ROI for the legend.
        Default: 'ROI'

    color : ndarray, optional
        RGB color for ROI.
        Default: np.array([0.9999, 0, 0])

    opacity : float, optional
        Opacity of ROI.
        Default: 1.0

    figure : Plotly Figure object, optional
        If provided, the visualization will be added to this Figure. Default:
        Initialize a new Figure.

    interact : bool
        Whether to open the visualization in an interactive window.
        Default: False

    inline : bool
        Whether to embed the interactive visualization inline in a notebook.
        Only works in the notebook context. Default: False.

    Returns
    -------
    Plotly Figure object
    """
    roi = vut.prepare_roi(roi, affine_or_mapping, static_img,
                          roi_affine, static_affine, reg_template)

    if figure is None:
        figure = go.Figure()

    set_layout(figure)

    _draw_roi(figure, roi, name, color, opacity)

    return _inline_interact(figure, interact, inline)


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

    v_min = volume.min()
    sf = volume.max() - v_min

    if axis == Axes.X:
        X, Y, Z = np.mgrid[height:height + 1,
                           :volume.shape[1], :volume.shape[2]]
        values = volume[height, :, :].flatten()
    elif axis == Axes.Y:
        X, Y, Z = np.mgrid[:volume.shape[0],
                           height:height + 1, :volume.shape[2]]
        values = volume[:, height, :].flatten()
    elif axis == Axes.Z:
        X, Y, Z = np.mgrid[:volume.shape[0],
                           :volume.shape[1], height:height + 1]
        values = volume[:, :, height].flatten()

    values = 1 - (values - v_min) / sf

    figure.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values,
            colorscale='greys',
            surface_count=1,
            showscale=False,
            opacity=opacity,
            visible=visible,
            name=_name_from_enum(axis),
            hoverinfo='skip'
        )
    )


def _name_from_enum(axis):
    if axis == Axes.X:
        return "Sagittal"
    elif axis == Axes.Y:
        return "Coronal"
    elif axis == Axes.Z:
        return "Axial"


def _draw_slices(figure, axis, volume,
                 opacity=0.3, sliders=[], n_steps=0, y_loc=0):
    if n_steps == 0:
        _draw_slice(figure, axis, volume, opacity=opacity)
    else:
        active = n_steps // 2
        name = _name_from_enum(axis) + " Plane"
        steps = []
        for step in range(n_steps):
            _draw_slice(figure, axis, volume, opacity=opacity,
                        step=step, n_steps=n_steps)

        for step in range(n_steps):
            step_dict = dict(
                method="update",
                args=[{"visible": [True] * len(figure.data)}],
                label=""
            )

            step_dict["args"][0]["visible"][-n_steps:] = [False] * n_steps
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
        ))


def visualize_volume(volume, figure=None, show_x=True, show_y=True,
                     show_z=True, interact=False, inline=False, opacity=0.3,
                     slider_definition=20, which_plane=None):
    """
    Visualize a volume

    Parameters
    ----------
    volume : ndarray or str
        3d volume to visualize.

    figure : Plotly Figure object, optional
        If provided, the visualization will be added to this Figure. Default:
        Initialize a new Figure.

    show_x : bool, optional
        Whether to show Coronal Slice.
        Default: True

    show_x : bool, optional
        Whether to show Sagittal Slice.
        Default: True

    show_x : bool, optional
        Whether to show Axial Slice.
        Default: True

    opacity : float, optional
        Opacity of slices.
        Default: 1.0

    slider_definition : int, optional
        How many discrete positions the slices can take.
        If 0, slices are stationary.
        Default: 50

    which_plane : str, optional
        Which plane can be moved with a slider.
        Should be 'Coronal', 'Axial', 'Sagittal', or None.
        If None, slices are stationary.
        Note: If slices are not stationary,
        do not add any more traces to the figure.
        Default: 'Coronal'

    interact : bool
        Whether to open the visualization in an interactive window.
        Default: False

    inline : bool
        Whether to embed the interactive visualization inline in a notebook.
        Only works in the notebook context. Default: False.

    Returns
    -------
    Plotly Figure object
    """
    volume = vut.load_volume(volume)

    if figure is None:
        figure = go.Figure()

    set_layout(figure)
    sliders = []

    # draw stationary slices first
    if show_x:
        if (which_plane is None) or which_plane.lower() != 'sagittal':
            _draw_slices(figure, Axes.X, volume, opacity=opacity, y_loc=0)
    if show_y:
        if (which_plane is None) or which_plane.lower() != 'coronal':
            _draw_slices(figure, Axes.Y, volume, opacity=opacity, y_loc=0)
    if show_z:
        if (which_plane is None) or which_plane.lower() != 'axial':
            _draw_slices(figure, Axes.Z, volume, opacity=opacity, y_loc=0)

    # Then draw interactive slices
    if show_x:
        if (which_plane is not None) and which_plane.lower() == 'sagittal':
            _draw_slices(figure, Axes.X, volume, sliders=sliders,
                         opacity=opacity, n_steps=slider_definition,
                         y_loc=0)
    if show_y:
        if (which_plane is not None) and which_plane.lower() == 'coronal':
            _draw_slices(figure, Axes.Y, volume, sliders=sliders,
                         opacity=opacity, n_steps=slider_definition,
                         y_loc=0)
    if show_z:
        if (which_plane is not None) and which_plane.lower() == 'axial':
            _draw_slices(figure, Axes.Z, volume, sliders=sliders,
                         opacity=opacity, n_steps=slider_definition,
                         y_loc=0)

    if slider_definition > 0 and which_plane is not None:
        figure.update_layout(sliders=tuple(sliders))

    return _inline_interact(figure, interact, inline)
