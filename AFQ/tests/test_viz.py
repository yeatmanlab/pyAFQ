import os.path as op

import nibabel.tmpdirs as nbtmp

import AFQ.data as afd
from AFQ import api


def test_plot_tract_profiles():
    tmpdir = nbtmp.InTemporaryDirectory()
    afd.organize_cfin_data(path=tmpdir.name)
    myafq = api.AFQ(dmriprep_path=op.join(tmpdir.name, 'cfin_multib',
                                          'derivatives', 'dmriprep'))
    myafq.plot_tract_profiles()
