
.. _methods_docs:

The pyAFQ API methods
---------------------
After defining your pyAFQ API object, you can ask for the output of
any step of the pipeline. It is common for users to just call export_all
(for example, `myafq.export_all()`). However, if the user only wants the
tractography, the user can instead call `myafq.export("streamlines")`. Here
is a list of all of pyAFQ's possible outputs:



data:
    DWI data as an ndarray for selected b values


gtab:
    A DIPY GradientTable with all the gradient information


img:
    unaltered DWI data in a Nifti1Image.


b0:
    full path to a nifti file containing the mean b0


masked_b0:
    full path to a nifti file containing the mean b0 after applying the brain mask


dti_tf:
    DTI TensorFit object


dti_params:
    full path to a nifti file containing parameters for the DTI fit


fwdti_tf:
    Free-water DTI TensorFit object


fwdti_params:
    Full path to a nifti file containing parameters for the free-water DTI fit.


dki_tf:
    DKI DiffusionKurtosisFit object


dki_params:
    full path to a nifti file containing parameters for the DKI fit


csd_params:
    full path to a nifti file containing parameters for the CSD fit


pmap:
    full path to a nifti file containing the anisotropic power map


fwdti_fa:
    full path to a nifti file containing the Free-water DTI fractional anisotropy


fwdti_md:
    full path to a nifti file containing the Free-water DTI mean diffusivity


fwdti_fwf:
    full path to a nifti file containing the Free-water DTI free water fraction


dti_fa:
    full path to a nifti file containing the DTI fractional anisotropy


dti_lt0:
    Image of first element in the DTI tensor


dti_lt1:
    Image of second element in the DTI tensor


dti_lt2:
    Image of third element in the DTI tensor


dti_lt3:
    Image of fourth element in the DTI tensor


dti_lt4:
    Image of fifth element in the DTI tensor


dti_lt5:
    Image of sixth element in the DTI tensor


dti_cfa:
    full path to a nifti file containing the DTI color fractional anisotropy


dti_pdd:
    full path to a nifti file containing the DTI principal diffusion direction


dti_md:
    full path to a nifti file containing the DTI mean diffusivity


dti_ga:
    full path to a nifti file containing the DTI geodesic anisotropy


dti_rd:
    full path to a nifti file containing the DTI radial diffusivity


dti_ad:
    full path to a nifti file containing the DTI axial diffusivity


dki_kt0:
    Image of first element in the DKI kurtosis model


dki_kt1:
    Image of second element in the DKI kurtosis model


dki_kt2:
    Image of third element in the DKI kurtosis model


dki_kt3:
    Image of fourth element in the DKI kurtosis model


dki_kt4:
    Image of fifth element in the DKI kurtosis model


dki_kt5:
    Image of sixth element in the DKI kurtosis model


dki_kt6:
    Image of seventh element in the DKI kurtosis model


dki_kt7:
    Image of eighth element in the DKI kurtosis model


dki_kt8:
    Image of ninth element in the DKI kurtosis model


dki_kt9:
    Image of tenth element in the DKI kurtosis model


dki_kt10:
    Image of eleventh element in the DKI kurtosis model


dki_kt11:
    Image of twelf element in the DKI kurtosis model


dki_kt12:
    Image of thirteenth element in the DKI kurtosis model


dki_kt13:
    Image of fourteenth element in the DKI kurtosis model


dki_kt14:
    Image of fifteenth element in the DKI kurtosis model


dki_lt0:
    Image of first element in the DTI tensor from DKI


dki_lt1:
    Image of second element in the DTI tensor from DKI


dki_lt2:
    Image of third element in the DTI tensor from DKI


dki_lt3:
    Image of fourth element in the DTI tensor from DKI


dki_lt4:
    Image of fifth element in the DTI tensor from DKI


dki_lt5:
    Image of sixth element in the DTI tensor from DKI


dki_fa:
    full path to a nifti file containing the DKI fractional anisotropy


dki_md:
    full path to a nifti file containing the DKI mean diffusivity


dki_awf:
    full path to a nifti file containing the DKI axonal water fraction


dki_mk:
    full path to a nifti file containing the DKI mean kurtosis file


dki_ga:
    full path to a nifti file containing the DKI geodesic anisotropy


dki_rd:
    full path to a nifti file containing the DKI radial diffusivity


dki_ad:
    full path to a nifti file containing the DKI axial diffusivity


dki_rk:
    full path to a nifti file containing the DKI radial kurtosis


dki_ak:
    full path to a nifti file containing the DKI axial kurtosis file


brain_mask:
    full path to a nifti file containing the brain mask


bundle_dict:
    Dictionary defining the different bundles to be segmented


reg_template:
    a Nifti1Image containing the template for registration


b0_warped:
    full path to a nifti file containing b0 transformed to template space


template_xform:
    full path to a nifti file containing registration template transformed to subject space


rois:
    dictionary of full paths to Nifti1Image files of ROIs transformed to subject space


mapping:
    mapping from subject to template space.


reg_subject:
    Nifti1Image which represents this subject when registering the subject to the template


bundles:
    full path to a trk file containing containting segmented streamlines, labeled by bundle


clean_bundles:
    full path to a trk file containting segmented streamlines, cleaned using the Mahalanobis distance, and labeled by bundle


indiv_bundles:
    dictionary of paths, where each path is a full path to a trk file containing the streamlines of a given bundle, cleaned or uncleaned


sl_counts:
    full path to a JSON file containing streamline counts


median_bundle_lengths:
    full path to a JSON file containing median bundle lengths


density_maps:
    full path to 4d nifti file containing streamline counts per voxel per bundle, where the 4th dimension encodes the bundle


profiles:
    full path to a CSV file containing tract profiles


scalar_dict:
    dicionary mapping scalar names to their respective file paths


seed:
    full path to a nifti file containing the tractography seed mask


stop:
    full path to a nifti file containing the tractography stop mask


streamlines:
    full path to the complete, unsegmented tractography file


all_bundles_figure:
    figure for the visualizaion of the recognized bundles in the subject's brain.


indiv_bundles_figures:
    list of full paths to html or gif files containing visualizaions of individual bundles


tract_profile_plots:
    list of full paths to png files, where files contain plots of the tract profiles


viz_backend:
    An instance of the `AFQ.viz.utils.viz_backend` class.
