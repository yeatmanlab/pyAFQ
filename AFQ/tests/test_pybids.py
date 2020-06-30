import AFQ.data as afd
from importlib import reload
reload(afd)
afd.organize_stanford_data()

from bids.layout import BIDSLayout

bids_dir = '/Users/arokem/AFQ_data/stanford_hardi'
deriv_dir = '/Users/arokem/AFQ_data/stanford_hardi/derivatives/dmriprep'

lay = BIDSLayout(bids_dir, derivatives=deriv_dir)