import tempfile
import os
import os.path as op
from collections import OrderedDict

import numpy as np
import IPython.display as display
import imageio as io
from palettable.tableau import Tableau_20

import nibabel as nib
from dipy.viz import window, actor, ui
from fury.colormap import line_colors

import AFQ.utils.volume as auv
import AFQ.registration as reg


tableau_20_rgb = np.array(Tableau_20.colors) / 255

color_dict = OrderedDict({"ATR_L": tableau_20_rgb[0],
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


def _inline_interact(scene, inline, interact):
    """
    Helper function to reuse across viz functions
    """
    if interact:
        window.show(scene)

    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        window.snapshot(scene, fname=fname, size=(1200, 1200))
        display.display_png(display.Image(fname))

    return scene


def visualize_bundles(trk, affine=None, bundle_dict=None, bundle=None,
                      colors=None, scene=None, background=(1, 1, 1),
                      interact=False, inline=False):
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

    interact : bool
        Whether to provide an interactive VTK window for interaction.
        Default: False

    inline : bool
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

    if scene is None:
        scene = window.Scene()

    scene.SetBackground(background[0], background[1], background[2])

    if colors is None:
        # Use the color dict provided
        colors = color_dict

    def _color_selector(bundle_dict, colors, b):
        """Helper function """
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
        return color

    if list(tg.data_per_streamline.keys()) == []:
        # There are no bundles in here:
        streamlines = list(streamlines)
        # Visualize all the streamlines with directionally assigned RGB:
        sl_actor = actor.line(streamlines, line_colors(streamlines))
        scene.add(sl_actor)
        sl_actor.GetProperty().SetRenderLinesAsTubes(1)
        sl_actor.GetProperty().SetLineWidth(6)

    else:
        # There are bundles:
        if bundle is None:
            # No selection: visualize all of them:
            for b in np.unique(tg.data_per_streamline['bundle']):
                idx = np.where(tg.data_per_streamline['bundle'] == b)[0]
                this_sl = list(streamlines[idx])
                color = _color_selector(bundle_dict, colors, b)
                sl_actor = actor.line(this_sl, color)
                sl_actor.GetProperty().SetRenderLinesAsTubes(1)
                sl_actor.GetProperty().SetLineWidth(6)
                scene.add(sl_actor)

        else:
            # Select just one to visualize:
            if isinstance(bundle, str):
                # We need to find the UID:
                uid = bundle_dict[bundle]['uid']
            else:
                # It's already a UID:
                uid = bundle

            idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
            this_sl = list(streamlines[idx])
            color = _color_selector(bundle_dict, colors, uid)
            sl_actor = actor.line(this_sl, color)
            sl_actor.GetProperty().SetRenderLinesAsTubes(1)
            sl_actor.GetProperty().SetLineWidth(6)
            scene.add(sl_actor)

    return _inline_interact(scene, inline, interact)


def scene_rotate_forward(scene):
    scene.elevation(90)
    scene.set_camera(view_up=(0.0, 0.0, 1.0))
    scene.reset_camera()
    return scene


def create_gif(scene, file_name, n_frames=60, size=(600, 600)):
    tdir = tempfile.gettempdir()
    window.record(scene, az_ang=360.0 / n_frames, n_frames=n_frames,
                  path_numbering=True, out_path=tdir + '/tgif',
                  size=size)

    angles = []
    for i in range(n_frames):
        if i < 10:
            angle_fname = f"tgif00000{i}.png"
        elif i < 100:
            angle_fname = f"tgif0000{i}.png"
        else:
            angle_fname = f"tgif000{i}.png"
        angles.append(io.imread(os.path.join(tdir, angle_fname)))

    io.mimsave(file_name, angles)


def visualize_roi(roi, affine_or_mapping=None, static_img=None,
                  roi_affine=None, static_affine=None, reg_template=None,
                  scene=None, color=np.array([1, 0, 0]), opacity=1.0,
                  inline=False, interact=False):
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

    if scene is None:
        scene = window.Scene()

    roi_actor = actor.contour_from_roi(roi, color=color, opacity=opacity)
    scene.add(roi_actor)

    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        window.snapshot(scene, fname=fname)
        display.display_png(display.Image(fname))

    return _inline_interact(scene, inline, interact)


def visualize_volume(volume, x=None, y=None, z=None, scene=None, inline=True,
                     interact=False):
    """
    Visualize a volume
    """
    if scene is None:
        scene = window.Scene()

    shape = volume.shape
    image_actor_z = actor.slicer(volume)
    slicer_opacity = 0.6
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    scene.add(image_actor_z)
    scene.add(image_actor_x)
    scene.add(image_actor_y)

    show_m = window.ShowManager(scene, size=(1200, 900))
    show_m.initialize()

    if interact:
        line_slider_z = ui.LineSlider2D(min_value=0,
                                        max_value=shape[2] - 1,
                                        initial_value=shape[2] / 2,
                                        text_template="{value:.0f}",
                                        length=140)

        line_slider_x = ui.LineSlider2D(min_value=0,
                                        max_value=shape[0] - 1,
                                        initial_value=shape[0] / 2,
                                        text_template="{value:.0f}",
                                        length=140)

        line_slider_y = ui.LineSlider2D(min_value=0,
                                        max_value=shape[1] - 1,
                                        initial_value=shape[1] / 2,
                                        text_template="{value:.0f}",
                                        length=140)

        opacity_slider = ui.LineSlider2D(min_value=0.0,
                                         max_value=1.0,
                                         initial_value=slicer_opacity,
                                         length=140)

        def change_slice_z(slider):
            z = int(np.round(slider.value))
            image_actor_z.display_extent(
                0, shape[0] - 1, 0, shape[1] - 1, z, z)

        def change_slice_x(slider):
            x = int(np.round(slider.value))
            image_actor_x.display_extent(
                x, x, 0, shape[1] - 1, 0, shape[2] - 1)

        def change_slice_y(slider):
            y = int(np.round(slider.value))
            image_actor_y.display_extent(
                0, shape[0] - 1, y, y, 0, shape[2] - 1)

        def change_opacity(slider):
            slicer_opacity = slider.value
            image_actor_z.opacity(slicer_opacity)
            image_actor_x.opacity(slicer_opacity)
            image_actor_y.opacity(slicer_opacity)

        line_slider_z.on_change = change_slice_z
        line_slider_x.on_change = change_slice_x
        line_slider_y.on_change = change_slice_y
        opacity_slider.on_change = change_opacity

        def build_label(text):
            label = ui.TextBlock2D()
            label.message = text
            label.font_size = 18
            label.font_family = 'Arial'
            label.justification = 'left'
            label.bold = False
            label.italic = False
            label.shadow = False
            label.background = (0, 0, 0)
            label.color = (1, 1, 1)

            return label

        line_slider_label_z = build_label(text="Z Slice")
        line_slider_label_x = build_label(text="X Slice")
        line_slider_label_y = build_label(text="Y Slice")
        opacity_slider_label = build_label(text="Opacity")

        panel = ui.Panel2D(size=(300, 200),
                           color=(1, 1, 1),
                           opacity=0.1,
                           align="right")
        panel.center = (1030, 120)

        panel.add_element(line_slider_label_x, (0.1, 0.75))
        panel.add_element(line_slider_x, (0.38, 0.75))
        panel.add_element(line_slider_label_y, (0.1, 0.55))
        panel.add_element(line_slider_y, (0.38, 0.55))
        panel.add_element(line_slider_label_z, (0.1, 0.35))
        panel.add_element(line_slider_z, (0.38, 0.35))
        panel.add_element(opacity_slider_label, (0.1, 0.15))
        panel.add_element(opacity_slider, (0.38, 0.15))

        show_m.scene.add(panel)

        global size
        size = scene.GetSize()

        def win_callback(obj, event):
            global size
            if size != obj.GetSize():
                size_old = size
                size = obj.GetSize()
                size_change = [size[0] - size_old[0], 0]
                panel.re_align(size_change)

    show_m.initialize()

    scene.zoom(1.5)
    scene.reset_clipping_range()

    if interact:
        show_m.add_window_callback(win_callback)
        show_m.render()
        show_m.start()

    return _inline_interact(scene, inline, interact)
