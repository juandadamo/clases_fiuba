
import numpy as np
import matplotlib.pyplot as plt
import scipy as Sci
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

T0=[370,293]
def func(T, t,k1,k2,k3):
	A=np.array(([-k1,k1],[k2,-k2]))
	B=np.array(([0,k3]))
	#print(A)
	return np.dot(A,T)+B


t = np.arange(0,2, 0.001)  
argf=(10.,45.,-20.)
argf=(9.,10.,-60.)
T = odeint(func, T0, t,argf)  

h=plt.plot(t,T,linewidth=2)
plt.legend(h,['$T_{huevos}$','$T_{agua}$'])
plt.axis([0,1,290,380])
plt.grid()
plt.show()
plt.savefig('tp.png')


#agregar conveccion natural alrededor de la esfera e integrar eso
#meter modelo transitorio mas complejo