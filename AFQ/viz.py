import tempfile
import os.path as op
import numpy as np
import IPython.display as display

import nibabel as nib
from dipy.viz.colormap import line_colors
from dipy.viz import window, actor, ui

from palettable.tableau import Tableau_20

import AFQ.utils.volume as auv
import AFQ.registration as reg

import pillow.Images as pilim


def _inline_interact(ren, inline, interact):
    """
    Helper function to reuse across viz functions
    """
    if interact:
        window.show(ren)

    if inline:
        tdir = tempfile.gettempdir()
        fname = op.join(tdir, "fig.png")
        window.record(ren, out_path=fname, size=(1200, 1200))
        display.display_png(display.Image(fname))

    return ren


def visualize_bundles(trk, affine_or_mapping=None, bundle=None, ren=None,
                      color=None, inline=True, interact=False):
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
        sl_actor.GetProperty().SetRenderLinesAsTubes(1)
        sl_actor.GetProperty().SetLineWidth(6)
    if bundle is None:
        for b in np.unique(tg.data_per_streamline['bundle']):
            idx = np.where(tg.data_per_streamline['bundle'] == b)[0]
            this_sl = list(streamlines[idx])
            if color is not None:
                sl_actor = actor.line(this_sl, color)
                sl_actor.GetProperty().SetRenderLinesAsTubes(1)
                sl_actor.GetProperty().SetLineWidth(6)
            else:
                sl_actor = actor.line(this_sl,
                                      Tableau_20.colors[np.mod(20, int(b))])
                sl_actor.GetProperty().SetRenderLinesAsTubes(1)
                sl_actor.GetProperty().SetLineWidth(6)

            ren.add(sl_actor)
    else:
        idx = np.where(tg.data_per_streamline['bundle'] == bundle)[0]
        this_sl = list(streamlines[idx])
        if color is not None:
            sl_actor = actor.line(this_sl, color)
            sl_actor.GetProperty().SetRenderLinesAsTubes(1)
            sl_actor.GetProperty().SetLineWidth(6)

        else:
            sl_actor = actor.line(
                this_sl,
                Tableau_20.colors[np.mod(20, int(bundle))])
            sl_actor.GetProperty().SetRenderLinesAsTubes(1)
            sl_actor.GetProperty().SetLineWidth(6)
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
    if ren is None:
        ren = window.ren()

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

    ren.add(image_actor_z)
    ren.add(image_actor_x)
    ren.add(image_actor_y)

    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()

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
        image_actor_z.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        image_actor_y.display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)

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

    show_m.ren.add(panel)

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel.re_align(size_change)

    show_m.initialize()

    ren.zoom(1.5)
    ren.reset_clipping_range()

    if interact:
        show_m.add_window_callback(win_callback)
        show_m.render()
        show_m.start()

    return _inline_interact(ren, inline, interact)


# TODO: Abstract this out to make simple render functions
# then pass the render into the save_spin functions
def save_spin_multibundle(bundle_list, Nframes=18,
                          savename='temp', showme=False,
                          savespin=True, size=(200, 200)):

    ren = window.Renderer()
    colormap = actor.create_colormap(np.arange(len(bundle_list) + 1))

    window.clear(ren)
    ren.set_camera(position=(-606.93, -153.23, 28.70),
                   focal_point=(2.78, 11.06, 15.66),
                   view_up=(0, 0, 1))

    ren.SetBackground(1, 1, 1)
    for i, clusters in enumerate(bundle_list):
        ren.add(actor.streamtube(clusters, colormap[i], linewidth=0.4))
    if savespin:
        increment = int(360 / Nframes)
        for i in range(0, Nframes):
            ren.yaw(increment)
            window.record(ren, out_path=savename + str(i) + '.png', size=size)
    else:
        window.record(ren, out_path=savename, size=size)
    if showme:
        window.show(ren)


def save_spin_single_bundle(bundle, Nframes=18,
                            savename='temp', showme=False,
                            savespin=True, size=(200, 200)):

    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)

    window.clear(ren)
    ren.set_camera(position=(-606.93, -153.23, 28.70),
                   focal_point=(2.78, 11.06, 15.66),
                   view_up=(0, 0, 1))

    ren.add(actor.streamtube(bundle, linewidth=0.4))

    if savespin:
        increment = int(360 / Nframes)
        for i in range(0, Nframes):
            ren.yaw(increment)
            window.record(ren, out_path=savename + str(i) + '.png', size=size)
    else:
        window.record(ren, out_path=savename, size=size)
    if showme:
        window.show(ren)


def make_mosaic(base_name):
    mosaic = pilim.new('RGB', (1000, 1000))
    hop = 0
    for j in range(0, 1000, 200):
        for i in range(0, 1000, 200):
            im = pilim.open(base_name + str(hop) + '.png')
            mosaic.paste(im, (i, j))
            hop += 1

    mosaic.show()
    mosaic.save(base_name + '_mosaic.png')
