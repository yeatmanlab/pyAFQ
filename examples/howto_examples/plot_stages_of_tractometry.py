"""
============================
The stages of tractometry
============================

This example visualizes the different stages of tractometry, from the
preprocessed diffusion data to the final tract profiles. We will use the Fury
library to visualize the results of pyAFQ in high-quality, publication-ready
figures.

"""

##############################################################################
# In this set of examples, we will use the `fury <https://fury.gl/>`_
# library to visualize outputs of pyAFQ as publication-ready figures.

import os
import os.path as op
import nibabel as nib
import numpy as np

from dipy.io.streamline import load_trk
from dipy.tracking.streamline import transform_streamlines, set_number_of_points
from dipy.core.gradients import gradient_table
from dipy.align import resample

from fury import actor, window
from fury.actor import colormap_lookup_table
from fury.colormap import create_colormap
from matplotlib.cm import tab20

import AFQ.data.fetch as afd
from AFQ.viz.utils import gen_color_dict


from PIL import Image
def make_video(frames, out):
    video = []
    for nn in frames:
        frame = Image.open(nn)
        video.append(frame)

    # Save the frames as an animated GIF
    video[0].save(
        out,
        save_all=True,
        append_images=video[1:],
        duration=300,
        loop=1)


#############################################################################
#
# .. note::
#   A virtual frame buffer is needed if you are running this example on
#   a machine that is not connected to a display ("headless"). If this is
#   the case, you can either run this example with the environment variable
#   "XVFB" set to "1" or "True" or you can remove the if statement below,
#   which will start a virtual frame buffer for you.

if os.environ.get("XVFB", False):
    print("Initializing XVFB")
    import xvfbwrapper
    from xvfbwrapper import Xvfb

    vdisplay = Xvfb()
    vdisplay.start()

###############################################################################
# Get some data from HBN POD2
# ----------------------------
# The Healthy Brain Network Preprocessed Open Diffusion Derivatives (HBN POD2)
# is a collection of resources based on the Healthy Brain Network dataset
# [1, 2]_. HBN POD2 includes data derivatives - including pyAFQ derivatives -
# from more than 2,000 subjects. The data and the derivatives can be browsed at
# https://fcp-indi.s3.amazonaws.com/index.html#data/Projects/HBN/BIDS_curated/
#
# Here, we will visualize the results from one subject, together with their
# anatomy and using several variations. We start by downloading their
# pyAFQ-processed data using fetcher functions that download both the
# preprocessed data, as well as the pyAFQ-processed data (Note that this
# will take up about 1.75 GB of disk space):

afd.fetch_hbn_preproc(["NDARAA948VFH"])
study_path = afd.fetch_hbn_afq(["NDARAA948VFH"])[1]

#############################################################################
# Visualize the processed dMRI data
# ---------------------------------
# The HBN POD2 dataset was processed using the ``qsiprep`` pipeline. The
# results from this processing are stored within a sub-folder of the
# derivatives folder wthin the study folder.
# Here, we will start by visualizing the diffusion data. We read in the
# diffusion data, as well as the gradient table, using the `nibabel` library.
# We then extract the b0, b1000, and b2000 volumes from the diffusion data.
# We will use the `actor.slicer` function from `fury` to visualize these. This
# function takes a 3D volume as input and returns a `slicer` actor, which can
# then be added to a `window.Scene` object. We create a helper function that
# will create a slicer actor for a given volume and a given slice along the x,
# y, or z dimension. We then call this function three times, once for each of
# the b0, b1000, and b2000 volumes, and add the resulting slicer actors to a
# scene. We set the camera on the scene to a view that we like, and then we
# record the scene into a png file. We do this for each of the three volumes.

deriv_path = op.join(
    study_path, "derivatives")

qsiprep_path = op.join(
    deriv_path,
    'qsiprep',
    'sub-NDARAA948VFH',
    'ses-HBNsiteRU')

dmri_img = nib.load(op.join(
        qsiprep_path,
        'dwi',
        'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.nii.gz'))

gtab = gradient_table(*[op.join(
        qsiprep_path,
        'dwi',
        f'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi.{ext}') for ext in ['bval', 'bvec']])


dmri_data = dmri_img.get_fdata()

dmri_b0 = dmri_data[..., 0]
dmri_b1000 = dmri_data[..., 1]
dmri_b2000 = dmri_data[..., 65]


def slice_volume(data, x=None, y=None, z=None):
    slicer_actors = []
    slicer_actor_z = actor.slicer(data)
    if z is not None:
        slicer_actor_z.display_extent(
            0, data.shape[0] - 1,
            0, data.shape[1] - 1,
            z, z)
        slicer_actors.append(slicer_actor_z)
    if y is not None:
        slicer_actor_y = slicer_actor_z.copy()
        slicer_actor_y.display_extent(
            0, data.shape[0] - 1,
            y, y,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_y)
    if x is not None:
        slicer_actor_x = slicer_actor_z.copy()
        slicer_actor_x.display_extent(
            x, x,
            0, data.shape[1] - 1,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_x)

    return slicer_actors

slicers_b0 = slice_volume(dmri_b0, x=dmri_b0.shape[0] // 2, y=dmri_b0.shape[1] // 2, z=dmri_b0.shape[-1] // 3)
slicers_b1000 = slice_volume(dmri_b1000, x=dmri_b0.shape[0] // 2, y=dmri_b0.shape[1] // 2, z=dmri_b0.shape[-1] // 3)
slicers_b2000 = slice_volume(dmri_b2000, x=dmri_b0.shape[0] // 2, y=dmri_b0.shape[1] // 2, z=dmri_b0.shape[-1] // 3)

for bval, slicers in zip([0, 1000, 2000], [slicers_b0, slicers_b1000, slicers_b2000]):
    scene = window.Scene()
    for slicer in slicers:
        scene.add(slicer)
    scene.set_camera(position=(721.34, 393.48, 97.03),
                     focal_point=(96.00, 114.00, 96.00),
                     view_up=(-0.01, 0.02, 1.00))

    scene.background((1, 1, 1))
    window.record(scene, out_path=f'b{bval}', size=(2400, 2400),
                  n_frames=36, path_numbering=True)

    make_video([f'b{bval}{ii:06d}.png' for ii in range(36)], f'b{bval}.gif')
#############################################################################
# Visualizing whole-brain tractography
# ------------------------------------
# One of the first steps of the pyAFQ pipeline is to generate whole-brain
# tractography. We will visualize the results of this step. We start by reading
# in the FA image, which is used as a reference for the tractography. We then

afq_path = op.join(
    deriv_path,
    'afq',
    'sub-NDARAA948VFH',
    'ses-HBNsiteRU')

fa_img = nib.load(op.join(afq_path,
                          'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_model-DKI_FA.nii.gz'))


sft_whole_brain = load_trk(op.join(afq_path,
                                   'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-probCSD_tractography.trk'), fa_img)


#############################################################################
# Transform into the T1w reference frame
# --------------------------------------
# Next, we would like to visualize the whole-brain tractography in the context
# of the T1w measurement. We read in this data and transform the bundle
# coordinates, first into the RASMM common coordinate frame and then
# subsequently into the coordinate frame of the T1-weighted data (if you find
# this confusing, you can brush up on this topic in the
# `nibabel documentation <https://nipy.org/nibabel/coordinate_systems.html>`_).


t1w_img = nib.load(op.join(deriv_path,
                           'qsiprep/sub-NDARAA948VFH/anat/sub-NDARAA948VFH_desc-preproc_T1w.nii.gz'))
t1w = t1w_img.get_fdata()
sft_whole_brain.to_rasmm()
whole_brain_t1w = transform_streamlines(
    sft_whole_brain.streamlines,
    np.linalg.inv(t1w_img.affine))


def lines_as_tubes(sl, line_width, **kwargs):
    line_actor = actor.line(sl, **kwargs)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor


whole_brain_actor = lines_as_tubes(whole_brain_t1w, 2)
slicers = slice_volume(t1w, x=t1w.shape[0] // 2, y=t1w.shape[1] // 2)

scene = window.Scene()

scene.add(whole_brain_actor)
for slicer in slicers:
    scene.add(slicer)

scene.set_camera(position=(721.34, 393.48, 97.03),
                 focal_point=(96.00, 114.00, 96.00),
                 view_up=(-0.01, 0.02, 1.00))

scene.background((1, 1, 1))
window.record(scene, out_path='whole_brain', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"whole_brain{ii:06d}.png" for ii in range(36)], "whole_brain.gif")

#############################################################################
# Whole brain with waypoints
# --------------------------------------
#

scene.clear()
whole_brain_actor = lines_as_tubes(whole_brain_t1w, 2)
slicers = slice_volume(t1w, x=t1w.shape[0] // 2, z=t1w.shape[-1] // 3)

scene.add(whole_brain_actor)
for slicer in slicers:
    scene.add(slicer)

scene.background((1, 1, 1))

waypoint1 = nib.load(
    op.join(
        afq_path,
        "ROIs", "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_desc-ROI-ARC_L-1-include.nii.gz"))

waypoint2 = nib.load(
    op.join(
        afq_path,
        "ROIs", "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_desc-ROI-ARC_L-2-include.nii.gz"))

waypoint1_xform = resample(waypoint1, t1w_img)
waypoint2_xform = resample(waypoint2, t1w_img)
waypoint1_data = waypoint1_xform.get_fdata() > 0
waypoint2_data = waypoint2_xform.get_fdata() > 0

surface_color = tab20.colors[0]

waypoint1_actor = actor.contour_from_roi(waypoint1_data,
                                         color=surface_color,
                                         opacity=0.5)

waypoint2_actor = actor.contour_from_roi(waypoint2_data,
                                         color=surface_color,
                                         opacity=0.5)

scene.add(waypoint1_actor)
scene.add(waypoint2_actor)

window.record(scene, out_path='whole_brain_with_waypoints', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"whole_brain_with_waypoints{ii:06d}.png" for ii in range(36)],
           "whole_brain_with_waypoints.gif")

bundle_path = op.join(afq_path,
                      'bundles')


#############################################################################
# Show the bundle
# ---------------
# The bundle coordinates from pyAFQ are always saved in the reference frame of
# the diffusion data from which they are generated, so we need an image file
# with the dMRI coordinates as a reference for loading the data (we could also
# use `"same"` here).

fa_img = nib.load(op.join(afq_path,
                          'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_model-DKI_FA.nii.gz'))
fa = fa_img.get_fdata()
sft_arc = load_trk(op.join(bundle_path,
                           'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq-ARC_L_tractography.trk'), fa_img)

sft_arc.to_rasmm()
arc_t1w = transform_streamlines(sft_arc.streamlines,
                                np.linalg.inv(t1w_img.affine))


bundles = [
    "ARC_R",
    "ATR_R",
    "CST_R",
    "IFO_R",
    "ILF_R",
    "SLF_R",
    "UNC_R",
    "CGC_R",
    "Orbital", "AntFrontal", "SupFrontal", "Motor",
    "SupParietal", "PostParietal", "Temporal", "Occipital",
    "CGC_L",
    "UNC_L",
    "SLF_L",
    "ILF_L",
    "IFO_L",
    "CST_L",
    "ATR_L",
    "ARC_L",
    ]

color_dict = gen_color_dict(bundles)

arc_actor = lines_as_tubes(arc_t1w, 8, colors=color_dict['ARC_L'])
scene.clear()

scene.add(arc_actor)
for slicer in slicers:
    scene.add(slicer)

scene.add(waypoint1_actor)
scene.add(waypoint2_actor)

window.record(scene, out_path='arc1', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"arc1{ii:06d}.png" for ii in range(36)], "arc1.gif")

#############################################################################
# Clean bundle
# ---------------

scene.clear()

scene.add(arc_actor)
for slicer in slicers:
    scene.add(slicer)

window.record(scene, out_path='arc2', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"arc2{ii:06d}.png" for ii in range(36)], "arc2.gif")

clean_bundles_path = op.join(afq_path,
                             'clean_bundles')

sft_arc = load_trk(op.join(clean_bundles_path,
                           'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq-ARC_L_tractography.trk'), fa_img)

sft_arc.to_rasmm()
arc_t1w = transform_streamlines(sft_arc.streamlines,
                                np.linalg.inv(t1w_img.affine))


arc_actor = lines_as_tubes(arc_t1w, 8, colors=tab20.colors[18])
scene.clear()

scene.add(arc_actor)
for slicer in slicers:
    scene.add(slicer)

window.record(scene, out_path='arc3', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"arc3{ii:06d}.png" for ii in range(36)], "arc3.gif")

#############################################################################
# Show the values of tissue properties along the bundle
# ---------------

lut_args = dict(scale_range=(0, 1),
                hue_range=(1, 0),
                saturation_range=(0, 1),
                value_range=(0, 1))

arc_actor = lines_as_tubes(arc_t1w, 8,
                           colors=resample(fa_img, t1w_img).get_fdata(),
                           lookup_colormap=colormap_lookup_table(**lut_args))
scene.clear()

scene.add(arc_actor)
for slicer in slicers:
    scene.add(slicer)

window.record(scene, out_path='arc4', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"arc4{ii:06d}.png" for ii in range(36)], "arc4.gif")

#############################################################################
# Core of the bundle and tract profile
# -------------------------------------

core_arc = np.median(np.asarray(set_number_of_points(arc_t1w, 20)), axis=0)

from dipy.stats.analysis import afq_profile
sft_arc.to_vox()
arc_profile = afq_profile(fa, sft_arc.streamlines, affine=np.eye(4),
                          n_points=20)

core_arc_actor = lines_as_tubes(
    [core_arc],
    40,
    colors=create_colormap(arc_profile, 'viridis')
)

arc_actor = lines_as_tubes(arc_t1w, 1,
                           colors=resample(fa_img, t1w_img).get_fdata(),
                           lookup_colormap=colormap_lookup_table(**lut_args))

scene.clear()

for slicer in slicers:
    scene.add(slicer)
scene.add(arc_actor)
scene.add(core_arc_actor)

window.record(scene, out_path='arc5', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"arc5{ii:06d}.png" for ii in range(36)], "arc5.gif")

scene.clear()

for slicer in slicers:
    scene.add(slicer)

for bundle in bundles:
    sft = load_trk(op.join(clean_bundles_path,
                            f'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq-{bundle}_tractography.trk'), fa_img)

    sft.to_rasmm()
    bundle_t1w = transform_streamlines(sft.streamlines,
                                       np.linalg.inv(t1w_img.affine))

    bundle_actor = lines_as_tubes(bundle_t1w, 8, colors=color_dict[bundle])
    scene.add(bundle_actor)

window.record(scene, out_path='all_bundles', size=(2400, 2400),
              n_frames=36, path_numbering=True)

make_video([f"all_bundles{ii:06d}.png" for ii in range(36)], "all_bundles.gif")


scene.clear()

for slicer in slicers:
    scene.add(slicer)

tract_profiles = []
for bundle in bundles:
    sft = load_trk(op.join(clean_bundles_path,
                            f'sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq-{bundle}_tractography.trk'), fa_img)
    sft.to_rasmm()
    bundle_t1w = transform_streamlines(sft.streamlines,
                                    np.linalg.inv(t1w_img.affine))

    core_bundle = np.median(np.asarray(set_number_of_points(bundle_t1w, 20)), axis=0)
    sft.to_vox()
    tract_profiles.append(
        afq_profile(fa, sft.streamlines, affine=np.eye(4),
                    n_points=20))

    core_actor = lines_as_tubes(
        [core_bundle],
        40,
        colors=create_colormap(tract_profiles[-1], 'viridis')
    )

    scene.add(core_actor)

window.record(scene,
              out_path='all_tract_profiles',
              size=(2400, 2400),
              n_frames=36,
              path_numbering=True)

make_video([f"all_tract_profiles{ii:06d}.png" for ii in range(36)],
            "all_tract_profiles.gif")

#############################################################################
# Tract profiles as a table

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for ii, bundle in enumerate(bundles):
    ax.plot(np.arange(ii * 20, (ii + 1) * 20),
            tract_profiles[ii],
            color=color_dict[bundle],
            linewidth=3)
ax.set_xticks(np.arange(0, 20 * len(bundles), 20))
ax.set_xticklabels(bundles, rotation=45, ha='right')
fig.set_size_inches(10, 4)
fig.savefig('tract_profiles_as_table.png')