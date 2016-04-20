import os.path as op
import boto3

import nibabel as nib
from dipy.data.fetcher import _make_fetcher

afq_home = op.join(op.expanduser('~'), 'AFQ_data')
baseurl = ("https://digital.lib.washington.edu/researchworks" +
           "/bitstream/handle/1773/34926/")

fnames = ["Callosum_midsag.nii.gz",
          "L_AntFrontal.nii.gz",
          "L_Motor.nii.gz",
          "L_Occipital.nii.gz",
          "L_Orbital.nii.gz",
          "L_PostParietal.nii.gz",
          "L_SupFrontal.nii.gz",
          "L_SupParietal.nii.gz",
          "L_Temporal.nii.gz",
          "R_AntFrontal.nii.gz",
          "R_Motor.nii.gz",
          "R_Occipital.nii.gz",
          "R_Orbital.nii.gz",
          "R_PostParietal.nii.gz",
          "R_SupFrontal.nii.gz",
          "R_SupParietal.nii.gz",
          "R_Temporal.nii.gz"]

md5_hashes = ["709fa90baadeacd64f1d62b5049a4125",
              "987c6169de807c4e93dc2cbd7a25d506",
              "0da114123d0b0097b96fe450a459550b",
              "6d845bd10504f67f1dc17f9000076d7e",
              "e16c7873ef4b08d26b77ef746dab8237",
              "47193fd4df1ea17367817466de798b90",
              "7e78bf9671e6945f4b2f5e7c30595a3c",
              "8adbb947377ff7b484c88d8c0ffc2125",
              "0fd981a4d0847e0642ff96e84fe44e47",
              "87c4855efa406d8fb004cffb8259180e",
              "c7969bcf5f2343fd9ce9c49b336cf14c",
              "bb4372b88991932150205ffb22aa6cb7",
              "d198d4e7db18ddc7236cf143ecb8342e",
              "d0f6edef64b0c710c92e634496085dda",
              "85eaee44665f244db5adae2e259833f6",
              "25f24eb22879a05d12bda007c81ea55a",
              "2664e0b8c2d9c59f13649a89bfcce399"]

fetch_templates = _make_fetcher("fetch_templates",
                                op.join(afq_home, 'templates'),
                                baseurl, fnames, fnames,
                                md5_list=md5_hashes,
                                doc="Download AFQ templates")


def read_templates():
    """Load AFQ templates from file

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    files, folder = fetch_templates()
    template_dict = {}
    for f in files:
        template_dict[f.split('.')[0]] = nib.load(op.join(folder, f))
    return template_dict


def fetch_hcp(subjects):
    """
    Fetch HCP diffusion data.

    Parameters
    ----------
    subjects : list
       Each item is an integer, identifying one of the HCP subjects

    Returns
    -------
    dict with remote and local names of thes files.

    Notes
    -----
    To use this function, you need to have a file '~/.aws/credentials', that
    includes a section:

    [hcp]
    AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
    AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX

    The keys are credentials that you can get from HCP (see https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)

    Local filenames are changed to match our expected conventions.
    """
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    base_dir = op.join(op.expanduser('~'), 'AFQ_data')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    data_files = {}
    for subject in subjects:
        sub_dir = op.join(base_dir, '%s' % subject)
        if not os.path.exists(op.join(base_dir, sub_dir)):
            os.mkdir(sub_dir)
            os.mkdir(os.path.join(sub_dir, 'sess'))
            os.mkdir(os.path.join(sub_dir, 'sess', 'dwi'))
            os.mkdir(os.path.join(sub_dir, 'sess', 'anat'))
        data_files[op.join(sub_dir, 'sess', 'dwi', 'dwi.bvals')] =\
            'HCP/%s/T1w/Diffusion/bvals' % subject
        data_files[op.join(sub_dir, 'sess', 'dwi', 'dwi.bvecs')] =\
            'HCP/%s/T1w/Diffusion/bvecs' % subject
        data_files[op.join(sub_dir, 'sess', 'dwi', 'dwi.nii.gz')] =\
            'HCP/%s/T1w/Diffusion/data.nii.gz' % subject
        data_files[op.join(sub_dir, 'sess', 'anat', 'T1w_acpc_dc.nii.gz')] =\
            'HCP/%s/T1w/T1w_acpc_dc.nii.gz' % subject

    for k in data_files.keys():
        if not op.exists(k):
            bucket.download_file(data_files[k], k)

    return data_files
