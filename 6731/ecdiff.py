
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
from scipy.integrate import odeint  
from scipy.special import gamma, airy  
y1_0 = 1.0/3**(2.0/3.0)/gamma(2.0/3.0)  
y0_0 = -1.0/3**(1.0/3.0)/gamma(1.0/3.0)  
y0 = [y0_0, y1_0]  
def func(y, t):
	return [t*y[1],y[0]]  

def gradient(y,t):
	return [[0,t],[1,0]]  

x = np.arange(0,4.0, 0.01)  
t = x  
ychk = airy(x)[0]  
y = odeint(func, y0, t)  
y2 = odeint(func, y0, t, Dfun=gradient)  

