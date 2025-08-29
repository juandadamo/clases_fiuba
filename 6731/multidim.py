import numpy as np
import scipy.special as ss
import matplotlib.pylab as plt
pi=np.pi


a=1
b=.5
T1=30.


x=np.linspace(0,a,50)
y=np.linspace(0,b,50)
mx,my=np.meshgrid(x,y)

sumat=[]
for n in range(1,100):
    sumat.append(2*(1-(-1)**n) / (np.pi*n*np.sinh(n*np.pi*b/a))*np.sin(n*np.pi*mx/a)*np.sinh(n*pi*my/a))
sumat=np.asarray(sumat)
T=np.sum(sumat,0)*T1

plt.contourf(x,y,T)
plt.contour(x,y,T,linewidth=14,colors='k')
plt.show()