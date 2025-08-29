
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
          'legend.fontsize': 20,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'text.usetex': True,
          'figure.figsize': (16,8)}
pylab.rcParams.update(params)
plt.close('all')
qv=100.
L=2.
la=1.
x=np.linspace(0,L,100)
Th=80 ; Tc= 30

for qv in np.linspace(0,100,7):
  T =  qv* L**2/( 2*la) * ( x/L-(x/L)**2 ) - (Th-Tc)/L *x +Th
  plt.plot(x,T,linewidth=3.,ms=7.,label='$\dot q_v=$ '+'%.0f' %(qv))
plt.xlabel('distancia [m]')
plt.ylabel('Temperatura [$^\circ C$] ')
plt.legend(loc=3,ncol=3,fancybox='True')
plt.grid('on')
plt.show()

#al=1.
#la1=50.
#la2=.08
#L=10.
#R2=0.09
#R1=.08

#Rc=la2/al
#r=np.linspace(R2+.0001,.4,200)
#Req= np.log(R2/R1) / ( la1*2*np.pi*L ) + np.log(r/R2) / ( la2*2*np.pi*L ) + 1/ ( 2*np.pi*L*al*r)
#plt.plot((r-R2)*100,Req,color='r',marker='s',linewidth=3.,ms=7.,label='$R2>Rc$')
#plt.xlabel('espesor de aislante [cm]')
#plt.ylabel('Resistencia global [W / K] ')
#plt.axis([0,25,0.16,.20])
#plt.grid('on')
#plt.legend(loc=4)
#plt.show()