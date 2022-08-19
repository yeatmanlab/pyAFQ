import tempfile
import enum
import logging

import numpy as np
import pandas as pd

import AFQ.viz.utils as vut

from dipy.tracking.streamline import set_number_of_points

try:
    import plotly
    import plotly.graph_objs as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import plotly.express as px
    from plotly.colors import hex_to_rgb
except (ImportError, ModuleNotFoundError):
    raise ImportError(vut.viz_import_msg_error("plotly"))


scope = pio.kaleido.scope
viz_logger = logging.getLogger("AFQ")


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
    if num <= 0.001:
        num = 0.001
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
        color = "rgba(0,0,0,0)"

    figure.update_layout(
        plot_bgcolor=color,
        scene1=dict(
            xaxis=dict(
                showbackground=False, showticklabels=False, title=''),
            yaxis=dict(
                showbackground=False, showticklabels=False, title=''),
            zaxis=dict(
                showbackground=False, showticklabels=False, title=''),
            aspectmode='data'
        )
    )


def _draw_streamlines(figure, sls, dimensions, color, name, cbv=None, cbs=None,
                      sbv_lims=[None, None], flip_axes=[False, False, False],
                      opacity=1.0):
    color = np.asarray(color)

    if len(sls._offsets) > 1:
        plotting_shape = (sls._data.shape[0] + sls._offsets.shape[0])
    else:
        plotting_shape = sls._data.shape[0]
    # dtype object so None can be stored
    x_pts = np.zeros(plotting_shape)
    y_pts = np.zeros(plotting_shape)
    z_pts = np.zeros(plotting_shape)

    if cbs is not None:
        cbs = np.asarray(cbs)
        line_color = np.zeros((plotting_shape, cbs.shape[1]))
        color = cbs[0, :]
    elif cbv is not None:
        if sbv_lims[0] is None:
            sbv_lims[0] = 0
        if sbv_lims[1] is None:
            sbv_lims[1] = cbv.max()

        color_constant = (color / color.max())\
            * (1.4 / (sbv_lims[1] - sbv_lims[0])) + sbv_lims[0]
        line_color = np.zeros((plotting_shape, 3))
    else:
        color_constant = color
        line_color = np.zeros((plotting_shape, 3))
    customdata = np.zeros(plotting_shape)

    for sl_index, plotting_offset in enumerate(sls._offsets):
        sl_length = sls._lengths[sl_index]
        sl = sls._data[plotting_offset:plotting_offset + sl_length]

        # add sl to lines
        total_offset = plotting_offset + sl_index
        x_pts[total_offset:total_offset + sl_length] = sl[:, 0]
        y_pts[total_offset:total_offset + sl_length] = sl[:, 1]
        z_pts[total_offset:total_offset + sl_length] = sl[:, 2]

        # don't draw between streamlines
        if len(sls._offsets) > 1:
            x_pts[total_offset + sl_length] = np.nan
            y_pts[total_offset + sl_length] = np.nan
            z_pts[total_offset + sl_length] = np.nan

        if cbs is not None:
            color_constant = cbs[sl_index]

        if cbv is not None:
            brightness = cbv[
                sl[:, 0].astype(int),
                sl[:, 1].astype(int),
                sl[:, 2].astype(int)
            ]

            line_color[total_offset:total_offset + sl_length, :] = \
                np.outer(brightness, color_constant)
            customdata[total_offset:total_offset + sl_length] = brightness
        else:
            line_color[total_offset:total_offset + sl_length, :] = \
                color_constant
            customdata[total_offset:total_offset + sl_length] = 1

        if line_color.shape[1] > 3:
            line_color[total_offset:total_offset + sl_length, 3] = \
                color_constant[3]  # dont shade alpha values

        if len(sls._offsets) > 1:
            line_color[total_offset + sl_length, :] = 0
            customdata[total_offset + sl_length] = 0

    if flip_axes[0]:
        x_pts = dimensions[0] - x_pts
    if flip_axes[1]:
        y_pts = dimensions[1] - y_pts
    if flip_axes[2]:
        z_pts = dimensions[2] - z_pts
    figure.add_trace(
        go.Scatter3d(
            x=x_pts,
            y=y_pts,
            z=z_pts,
            name=vut.display_string(name),
            legendgroup=vut.display_string(name),
            marker=dict(
                size=0.0001,
                color=_color_arr2str(color)
            ),  # this is necessary to add color to legend
            line=dict(
                width=8,
                color=line_color,
            ),
            hovertext=customdata,
            hoverinfo='all',
            opacity=opacity
        ),
        row=1, col=1
    )
    return color_constant


def _plot_profiles(profiles, bundle_name, color, fig, scalar):
    if isinstance(profiles, pd.DataFrame):
        profiles = profiles[profiles.tractID == bundle_name]
        x = profiles["nodeID"]
        y = profiles[scalar]
        line_color = []
        for scalar_val in profiles[scalar].to_numpy():
            line_color.append(_color_arr2str(scalar_val * color))
    else:
        x = np.arange(len(profiles))
        y = profiles
        line_color = []
        for indiv_color in color:
            line_color.append(_color_arr2str(indiv_color))

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=np.zeros(len(y)),
            name=vut.display_string(bundle_name),
            line=dict(color=line_color, width=15),
            mode="lines",
            legendgroup=vut.display_string(bundle_name)),
        row=1, col=2)

    font = dict(size=20, family="Overpass")
    fixed_camera_for_2d = dict(
        projection=dict(type="orthographic"),
        up=dict(x=0, y=1, z=0),
        eye=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0))
    fig.update_layout(
        margin={"t": 15, "b": 0, "l": 0, "r": 0},
        scene2=dict(
            camera=fixed_camera_for_2d,
            zaxis=dict(visible=False),
            dragmode=False,
            xaxis_title=dict(text="Location", font=font),
            yaxis_title=dict(text=vut.display_string(scalar), font=font)))


def visualize_bundles(sft, n_points=None, bundle_dict=None,
                      bundle=None, colors=None, shade_by_volume=None,
                      color_by_streamline=None,
                      sbv_lims=[None, None], include_profiles=(None, None),
                      flip_axes=[False, False, False], opacity=1.0,
                      figure=None, background=(1, 1, 1), interact=False,
                      inline=False):
    """
    Visualize bundles in 3D

    Parameters
    ----------
    sft : Stateful Tractogram, str
        A Stateful Tractogram containing streamline information
        or a path to a trk file.
        In order to visualize individual bundles, the Stateful Tractogram
        must contain a bundle key in it's data_per_streamline which is a list
        of bundle `'uid'`.

    n_points : int or None
        n_points to resample streamlines to before plotting. If None, no
        resampling is done.

    bundle_dict : dict, optional
        Keys are names of bundles and values are dicts that specify them.
        Default: bundles are either not identified, or identified
        only as unique integers in the metadata.

    bundle : str, optional
        The name of a bundle to select from among the keys in `bundle_dict`
        or an integer for selection from the sft metadata.

    colors : dict or list
        If this is a dict, keys are bundle names and values are RGB tuples.
        If this is a list, each item is an RGB tuple. Defaults to a list
        with Tableau 20 RGB values if bundle_dict is None, or dict from
        bundles to Tableau 20 RGB values if bundle_dict is not None.

    shade_by_volume : ndarray or str, optional
        3d volume use to shade the bundles. If None, no shading
        is performed. Only works when using the plotly backend.
        Default: None

    color_by_streamline : ndarray or dict, optional
        N by 3 array, where N is the number of streamlines in sft;
        for each streamline you specify 3 values between 0 and 1 for rgb.
        If sft has multiple bundles, then use a dict for color_by_streamline,
        where keys are bundle names and values are n by 3 arrays.
        Overrides colors for bundles in the keys
        of the dict if passing a  dict, or for all streamlines if using
        ndarray.
        Default: None

    sbv_lims : ndarray
        Of the form (lower bound, upper bound). Shading based on
        shade_by_volume will only differentiate values within these bounds.
        If lower bound is None, will default to 0.
        If upper bound is None, will default to the maximum value in
        shade_by_volume.
        Default: [None, None]

    include_profiles : Tuple of Pandas Dataframe and string
        The first element of the uple is a
        Pandas Dataframe containing profiles in the standard pyAFQ
        output format for the bundle(s) being displayed. It will be used
        to generate a graph of the tract profiles for each bundle,
        with colors corresponding to the colors on the bundles. The string
        is the scalar to use from the profile. If these are None,
        no tract profiles will be graphed.
        Defualt: (None, None)

    flip_axes : ndarray
        Which axes to flip, to orient the image as RAS, which is how we
        visualize.
        For example, if the input image is LAS, use [True, False, False].
        Default: [False, False, False]

    opacity : float
        Float between 0 and 1 defining the opacity of the bundle.
        Default: 1.0

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

    if shade_by_volume is not None:
        shade_by_volume = vut.load_volume(shade_by_volume)

    if figure is None:
        if include_profiles[0] is None:
            figure = make_subplots(
                rows=1, cols=1,
                specs=[[{"type": "scene"}]])
        else:
            figure = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}]])

    set_layout(figure, color=_color_arr2str(background))

    for (sls, color, name, dimensions) in vut.tract_generator(
            sft, bundle, bundle_dict, colors, n_points):
        if isinstance(color_by_streamline, dict):
            if name in color_by_streamline:
                cbs = color_by_streamline[name]
        else:
            cbs = color_by_streamline

        color_constant = _draw_streamlines(
            figure,
            sls,
            dimensions,
            color,
            name,
            cbv=shade_by_volume,
            cbs=cbs,
            sbv_lims=sbv_lims,
            flip_axes=flip_axes,
            opacity=opacity)
        if include_profiles[0] is not None:
            _plot_profiles(
                include_profiles[0], name, color_constant,
                figure, include_profiles[1])

    figure.update_layout(legend=dict(itemsizing="constant"))
    return _inline_interact(figure, interact, inline)


def create_gif(figure,
               file_name,
               n_frames=30,
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
        scope._shutdown_kaleido()  # temporary fix for memory leak

    vut.gif_from_pngs(tdir, file_name, n_frames,
                      png_fname="tgif", add_zeros=False)


def _draw_roi(figure, roi, name, color, opacity, dimensions, flip_axes):
    roi = np.where(roi == 1)
    pts = []
    for i, flip in enumerate(flip_axes):
        if flip:
            pts.append(dimensions[i] - (roi[i] + 1))
        else:
            pts.append(roi[i] + 1)
    figure.add_trace(
        go.Scatter3d(
            x=pts[0],
            y=pts[1],
            z=pts[2],
            name=name,
            marker=dict(color=_color_arr2str(color, opacity=opacity)),
            line=dict(color="rgba(0,0,0,0)")
        ),
        row=1, col=1
    )


def visualize_roi(roi, affine_or_mapping=None, static_img=None,
                  roi_affine=None, static_affine=None, reg_template=None,
                  name='ROI', figure=None, flip_axes=[False, False, False],
                  color=np.array([0.9999, 0, 0]),
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

    flip_axes : ndarray
        Which axes to flip, to orient the image as RAS, which is how we
        visualize.
        For example, if the input image is LAS, use [True, False, False].
        Default: [False, False, False]

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
        figure = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "scene"}]])

    set_layout(figure)

    _draw_roi(figure, roi, name, color, opacity, roi.shape, flip_axes)

    return _inline_interact(figure, interact, inline)


class Axes(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


def _draw_slice(figure, axis, volume, opacity=0.3, pos=0.5,
                colorscale="greys", invert_colorscale=False):
    height = int(volume.shape[axis] * pos)

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

    values = (values - v_min) / sf
    if invert_colorscale:
        values = 1 - values

    figure.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values,
            colorscale=colorscale,
            surface_count=1,
            showscale=False,
            opacity=opacity,
            name=_name_from_enum(axis),
            hoverinfo='skip',
            showlegend=True
        ),
        row=1, col=1
    )


def _name_from_enum(axis):
    if axis == Axes.X:
        return "Sagittal"
    elif axis == Axes.Y:
        return "Coronal"
    elif axis == Axes.Z:
        return "Axial"


def visualize_volume(volume, figure=None, x_pos=0.5, y_pos=0.5,
                     z_pos=0.5, interact=False, inline=False, opacity=0.3,
                     colorscale="gray", invert_colorscale=False,
                     flip_axes=[False, False, False]):
    """
    Visualize a volume

    Parameters
    ----------
    volume : ndarray or str
        3d volume to visualize.

    figure : Plotly Figure object, optional
        If provided, the visualization will be added to this Figure. Default:
        Initialize a new Figure.

    x_pos : float or None, optional
        Where to show Coronal Slice. If None, slice is not shown.
        Should be a decimal between 0 and 1.
        Indicatesthe fractional position along the perpendicular
        axis to the slice.
        Default: 0.5

    y_pos : float or None, optional
        Where to show Sagittal Slice. If None, slice is not shown.
        Should be a decimal between 0 and 1.
        Indicatesthe fractional position along the perpendicular
        axis to the slice.
        Default: 0.5

    z_pos : float or None, optional
        Where to show Axial Slice. If None, slice is not shown.
        Should be a decimal between 0 and 1.
        Indicatesthe fractional position along the perpendicular
        axis to the slice.
        Default: 0.5

    opacity : float, optional
        Opacity of slices.
        Default: 1.0

    colorscale : string, optional
        Plotly colorscale to use to color slices.
        Default: "greys"

    invert_colorscale : bool, optional
        Whether to invert colorscale.
        Default: False

    flip_axes : ndarray
        Which axes to flip, to orient the image as RAS, which is how we
        visualize.
        For example, if the input image is LAS, use [True, False, False].
        Default: [False, False, False]

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
    for i, flip in enumerate(flip_axes):
        if flip:
            volume = np.flip(volume, axis=i)

    if figure is None:
        figure = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "scene"}]])

    set_layout(figure)

    for pos, axis in [(x_pos, Axes.X), (y_pos, Axes.Y), (z_pos, Axes.Z)]:
        if pos is not None:
            _draw_slice(
                figure, axis, volume, opacity=opacity,
                pos=pos, colorscale=colorscale,
                invert_colorscale=invert_colorscale)

    return _inline_interact(figure, interact, inline)


def _draw_core(sls, n_points, figure, bundle_name, indiv_profile,
               labelled_points, dimensions, flip_axes):
    fgarray = np.asarray(set_number_of_points(sls, n_points))
    fgarray = np.median(fgarray, axis=0)
    # colormap = px.colors.diverging.Portland
    # colormap = np.asarray(
    #     [[int(i) for i in c[4:-1].split(',')] for c in colormap]) / 256
    colormap = px.colors.sequential.Viridis
    colormap = np.asarray(
        [hex_to_rgb(c) for c in colormap]) / 256
    xp = np.linspace(
        np.min(indiv_profile),
        np.max(indiv_profile),
        num=len(colormap))
    line_color = np.ones((n_points, 4))
    for i in range(3):
        line_color[:, i] = np.interp(indiv_profile, xp, colormap[:, i])
    line_color_untouched = line_color.copy()
    for i in range(n_points):
        if i < n_points - 1:
            direc = fgarray[i + 1] - fgarray[i]
            direc = direc / np.linalg.norm(direc)
            light_direc = -fgarray[i] / np.linalg.norm(fgarray[i])
            direc_adjust = np.dot(direc, light_direc)
            direc_adjust = (direc_adjust + 3) / 4
        line_color[i, 0:3] = line_color[i, 0:3] * direc_adjust
    text = [None] * n_points
    for label in labelled_points:
        if label == -1:
            text[label] = str(n_points)
        else:
            text[label] = str(label)

    if flip_axes[0]:
        fgarray[:, 0] = dimensions[0] - fgarray[:, 0]
    if flip_axes[1]:
        fgarray[:, 1] = dimensions[1] - fgarray[:, 1]
    if flip_axes[2]:
        fgarray[:, 2] = dimensions[2] - fgarray[:, 2]

    figure.add_trace(
        go.Scatter3d(
            x=fgarray[:, 0],
            y=fgarray[:, 1],
            z=fgarray[:, 2],
            name=vut.display_string(bundle_name + "_core"),
            line=dict(
                width=25,
                color=line_color,
            ),
            hovertext=indiv_profile,
            hoverinfo='all',
            text=text,
            textfont=dict(size=20, family="Overpass"),
            textposition="top right",
            mode="lines+text"
        ),
        row=1, col=1
    )

    return line_color_untouched


def single_bundle_viz(indiv_profile, sft,
                      bundle, scalar_name,
                      bundle_dict=None,
                      flip_axes=[False, False, False],
                      labelled_nodes=[0, -1],
                      figure=None,
                      include_profile=False):
    """
    Visualize a single bundle in 3D with core bundle and associated profile

    Parameters
    ----------
    indiv_profile : ndarray
        A numpy array containing a tract profile for this bundle for a scalar.

    sft : Stateful Tractogram, str
        A Stateful Tractogram containing streamline information.
        If bundle is an int, the Stateful Tractogram
        must contain a bundle key in it's data_per_streamline which is a list
        of bundle `'uid'.
        Otherwise, the entire Stateful Tractogram will be used as the bundle
        for the visualization.

    bundle : str or int
        The name of the bundle to be used as the label for the plot,
        and for selection from the sft metadata.

    scalar_name : str
        The name of the scalar being used.

    bundle_dict : dict, optional
        This parameter is used if bundle is an int.
        Keys are names of bundles and values are dicts that specify them.
        Default: Either the entire sft is treated as a bundle,
        or identified only as unique integers in the metadata.

    flip_axes : ndarray
        Which axes to flip, to orient the image as RAS, which is how we
        visualize.
        For example, if the input image is LAS, use [True, False, False].
        Default: [False, False, False]

    labelled_nodes : list or ndarray
        Which nodes to label. -1 indicates the last node.
        Default: [0, -1]

    figure : Plotly Figure object, optional
        If provided, the visualization will be added to this Figure. Default:
        Initialize a new Figure.

    include_profile : bool, optional
        If true, also plot the tract profile. Default: False

    Returns
    -------
    Plotly Figure object
    """
    if figure is None:
        if include_profile:
            figure = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}]])
        else:
            figure = make_subplots(
                rows=1, cols=1,
                specs=[[{"type": "scene"}]])

    set_layout(figure)

    n_points = len(indiv_profile)
    sls, _, bundle_name, dimensions = next(vut.tract_generator(
        sft, bundle, bundle_dict, None, n_points))

    line_color = _draw_core(
        sls, n_points, figure, bundle_name, indiv_profile,
        labelled_nodes, dimensions, flip_axes)

    if include_profile:
        _plot_profiles(
            indiv_profile, bundle_name + "_profile",
            line_color, figure, scalar_name)

    return figure
