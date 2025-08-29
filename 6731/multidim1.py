import numpy as np
import scipy.special as ss
import matplotlib.pylab as plt
import pylab 
params = {'backend': 'ps',
'axes.labelsize': 26,
'text.fontsize': 16,
'legend.fontsize': 20,
'xtick.labelsize': 28,
'ytick.labelsize': 28,
'text.usetex': True,
'figure.figsize': (12,10)}
pylab.rcParams.update(params)
plt.close('all')
pi=np.pi

#a=2.12e-5
a=0.025
b=a
T1=40.


x=np.linspace(0,a,50)
y=np.linspace(0,b,50)
mx,my=np.meshgrid(x,y)

sumat=[]
for n in range(1,100):
    sumat.append(2*(1-(-1)**n) / (np.pi*n*np.sinh(n*np.pi*b/a))*np.sin(n*np.pi*mx/a)*np.sinh(n*pi*my/a))
sumat=np.asarray(sumat)
T=np.sum(sumat,0)*T1
fig1,ax1=plt.subplots(1,1)
cm1=ax1.contourf(x,y,T,levels=np.linspace(0,T1,10))
fig1.colorbar(cm1,format='%.2f')
ax1.contour(x,y,T,linewidths=2,colors='k',levels=np.linspace(0,T1,10),inline=1)
#ax1.xlabel('$x$')
#plt.ylabel('$y$')
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
plt.savefig('/home/juan/Documents/Ensenanza/latex/conduccion_transitorio/isotermas.pdf')
plt.show()
