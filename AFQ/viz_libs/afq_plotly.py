import tempfile
import os
import os.path as op
import enum

import numpy as np
import plotly
import plotly.graph_objs as go

import dipy.tracking.streamlinespeed as dps

import AFQ.viz_libs.utils as vut


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
                      resample=100, interact=False, inline=False):
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

    tg, streamlines = vut.tract_loader(trk, affine)

    if figure is None:
        figure = go.Figure()

    figure.update_layout(plot_bgcolor=_color_arr2str(background))

    for (sls, color, name) in \
            vut.tract_generator(tg, streamlines, bundle, bundle_dict, colors):
        _draw_streamlines(figure, sls, color, name, n_points=resample)

    return _inline_interact(figure, interact, inline)


def stop_creating_gifs():
    try:
        plotly.io.orca.shutdown_server()
    except ValueError:
        _orca_err()


def _orca_err():
    print("pyAFQ is trying to generate gifs using plotly. "
          + "This requires orca, which cannot be installed via pip. "
          + "See: https://github.com/plotly/orca \n"
          + "Or consider using fury to visualize with pyAFQ instead")


def create_gif(figure,
               file_name,
               n_frames=60,
               zoom=2.5,
               z_offset=0.5,
               creating_many=False,
               size=(600, 600),
               rotate_forward=True):
    tdir = tempfile.gettempdir()

    for i in range(n_frames):
        theta = (i * 6.28) / n_frames
        camera = dict(
            eye=dict(x=np.cos(theta) * zoom,
                     y=np.sin(theta) * zoom, z=z_offset)
        )
        figure.update_layout(scene_camera=camera)
        try:
            figure.write_image(tdir + f"/tgif{i}.png")
        except ValueError:
            _orca_err()
            return

    if not creating_many:
        stop_creating_gifs()

    vut.gif_from_pngs(tdir, file_name, n_frames,
                      png_fname="tgif", add_zeros=False)


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
                  interact=False, inline=False):
    """
    Render a region of interest into a VTK viz as a volume
    """
    roi = vut.prepare_roi(roi, affine_or_mapping, static_img,
                          roi_affine, static_affine, reg_template)

    if figure is None:
        figure = go.Figure()

    figure.update_layout(plot_bgcolor=f"rgba(0,0,0,0)")

    _draw_roi(figure, roi, color, opacity)

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


def visualize_volume(volume, figure=None, show_x=True, show_y=True,
                     show_z=True, interact=False, inline=False, opacity=0.3,
                     slider_definition=0):
    """
    Visualize a volume
    """
    volume = vut.load_volume(volume)

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

    return _inline_interact(figure, interact, inline)
