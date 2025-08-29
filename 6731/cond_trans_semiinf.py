
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 12:29:58 2015
"""
from matplotlib.patches import Ellipse, Arc
from matplotlib import cm, colors, patches, markers
from math import pi
from matplotlib.mlab import griddata
from  scipy import optimize
import glob,os,sys
import matplotlib.pyplot as plt
import pylab,six
import numpy as np
from scipy.optimize import fmin
import matplotlib.animation as animation
from scipy import integrate
from matplotlib import animation
mk=np.tile(markers.MarkerStyle.filled_markers,3)
ck=list(six.iteritems(colors.cnames))
from scipy import integrate
from scipy.special import erf

plt.close('all')

x= np.linspace(0,1.5,500)
#plt.plot(x,(1-erf(x/(4.*1)**.5))-np.exp(5*x+5**2)*(1-erf(x/(4.*1)**.5+5*1**.5))) 
#plt.plot(x,(1-erf(x/(4.*1)**.5))-np.exp(5*x+5**2)*(1-erf(x/(4.*1)**.5+5*1**.5))) 
c=5.
fig,ax = plt.subplots()
for t in [.25,.5,.9,2]:
  ax.plot(x,(1-erf(x/(4*t)**.5))-np.exp(x*c+c**2*t)*(1-erf(x/(4*t)**.5+c*t**.5) ))

plt.show()


exp =np.exp
plt.plot(x,(1-erf(x/(4*.9)**.5) ) - exp(5*x+25*.9)*(1-erf(x/(4*.9)**.5+5*.9**.5)) ,'ro')

plt.plot(x,(1-erf(x/(4*.23)**.5) ) - exp(5*x+25*.23)*(1-erf(x/(4*.23)**.5+5*.23**.5)) ,'bd')