import os.path as op

import nibabel.tmpdirs as nbtmp

import AFQ.data as afd
from AFQ import api


def test_plot_tract_profiles():
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_stanford_data(path=tmpdir.name)
    myafq = api.AFQ(dmriprep_path=op.join(tmpdir.name, 'stanford_hardi',
                                          'derivatives', 'dmriprep'),
                    sub_prefix='sub')
    myafq.plot_tract_profiles()
