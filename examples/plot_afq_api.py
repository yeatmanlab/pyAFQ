"""
==========================
AFQ API
==========================

An example using the AFQ API

"""

import matplotlib.pyplot as plt
import os.path as op

from AFQ import api
import AFQ.data as afd

afd.organize_stanford_data()
base_dir = op.join(op.expanduser('~'), 'AFQ_data', 'stanford_hardi')
myafq = api.AFQ(preproc_path=op.join(afd.afq_home, 'stanford_hardi'),
                sub_prefix='sub')

fig, ax = plt.subplots(1)
ax.matshow(myafq.dti[0].fa[:, :, 40])
ax.axis("off")
