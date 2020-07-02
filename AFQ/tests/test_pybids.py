import AFQ.data as afd
from importlib import reload
reload(afd)
afd.organize_stanford_data()

from bids.layout import BIDSLayout

bids_dir = '/Users/arokem/AFQ_data/stanford_hardi'

lay = BIDSLayout(bids_dir, derivatives=True)

dwi_sub_01 = lay.get(subject='01', extension='nii.gz', suffix='dwi', return_type='filename', scope='vistasoft')
bvals_sub_01 = lay.get(subject='01', extension='bvals', suffix='dwi', return_type='filename', scope='vistasoft')
bvecs_sub_01 = lay.get(subject='01', extension='bvecs', suffix='dwi', return_type='filename', scope='vistasoft')
t1w_sub_01 = lay.get(subject='01', extension='.nii.gz', suffix='T1w', return_type='filename')
seg_sub_01 = lay.get(subject='01', extension='.nii.gz', suffix='seg', return_type='filename')