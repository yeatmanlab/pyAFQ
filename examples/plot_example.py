"""
=============
Plotting Data
=============

An example plot of `

"""
import numpy as np
from matplotlib import pyplot as plt

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
plt.plot(X, y)
plt.show()
