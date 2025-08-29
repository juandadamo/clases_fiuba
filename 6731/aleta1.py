
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
tf=20
tb=80
th0=tb-tf
L=3e-2
x=np.linspace(0,L,100)
R=0.005
al=3
la=10
P=2*np.pi*R
St=np.pi*R**2


for al in np.linspace(.5,20,7):
  beta=al *P / (la * St)
  #th=th0* ( np.exp(beta*x) / (1+np.exp(2*beta*L) )  + np.exp(-beta*x) / (1-np.exp(-2*beta*L) ) ) 
  th2=th0*np.cosh(beta*(L-x))/np.cosh(beta*L)
#plt.plot(x,th)
  plt.plot(x/L,th2/th0,linewidth=3.,label='$\chi=$ '+'%.0f' %(beta*L))
plt.xlabel('x /L')
plt.ylabel('$\theta/\theta_0$ ')
plt.grid('on')
plt.legend()
plt.show()
#\theta=\theta_0 \left[  \frac{e^{\beta x}}{1+e^{2\beta L}} +  \frac{e^{-\beta x}}{1-e^{-2\beta
#L}}\right]=\theta_0\frac{\cosh(\beta(L-x))}{\cosh(\beta L)}















#qv=100.
#L=2.
#la=1.
#x=np.linspace(0,L,100)
#Th=80 ; Tc= 30

#for qv in np.linspace(0,100,7):
  #T =  qv* L**2/( 2*la) * ( x/L-(x/L)**2 ) - (Th-Tc)/L *x +Th
  #plt.plot(x,T,linewidth=3.,ms=7.,label='$\dot q_v=$ '+'%.0f' %(qv))
#plt.xlabel('distancia [m]')
#plt.ylabel('Temperatura [$^\circ C$] ')
#plt.legend(loc=3,ncol=3,fancybox='True')
#plt.grid('on')
#plt.show()

##al=1.
##la1=50.
##la2=.08
##L=10.
##R2=0.09
##R1=.08

##Rc=la2/al
##r=np.linspace(R2+.0001,.4,200)
##Req= np.log(R2/R1) / ( la1*2*np.pi*L ) + np.log(r/R2) / ( la2*2*np.pi*L ) + 1/ ( 2*np.pi*L*al*r)
##plt.plot((r-R2)*100,Req,color='r',marker='s',linewidth=3.,ms=7.,label='$R2>Rc$')
##plt.xlabel('espesor de aislante [cm]')
##plt.ylabel('Resistencia global [W / K] ')
##plt.axis([0,25,0.16,.20])
##plt.grid('on')
##plt.legend(loc=4)
##plt.show()
