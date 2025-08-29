
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy as Sci
import time
import glob
import pylab 
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'text.fontsize': 16,
          'legend.fontsize': 18,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'text.usetex': True,
          'figure.figsize': (10,8)}
pylab.rcParams.update(params)


t=np.linspace(0,10,200)
plt.plot(t,np.exp(-t*.8),'k',linewidth=2,label='solido temp. uniforme')
plt.ylabel('$(T-T_\infty) / (T_o - T_\infty)$')
plt.xlabel('$t$')
plt.grid()
plt.legend()
