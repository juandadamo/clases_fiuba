import numpy as np
import scipy.special as ss
import matplotlib.pylab as plt
import pylab
params = {'backend': 'ps',
'axes.labelsize': 26,
'text.fontsize': 16,
'legend.fontsize': 18,
'xtick.labelsize': 28,
'ytick.labelsize': 28,
'text.usetex': True,
'figure.figsize': (8,8),
'image.cmap':'OrRd'}
pylab.rcParams.update(params)


mk=np.tile(['o','s','<','D','>','h','p','x','d','^'],3)
ck=np.tile(['k','r','b','g','m','y','c','k'],3)
a=25e-3
Ts=150.
Tini=10.
plt.close('all')
t=np.linspace(0.01,1,4)
x=np.linspace(0,.3,100)
T=np.asarray([Ts+(Tini-Ts)*ss.erf( x / (2*(a*t[i])**(1/2.))) for i in range(len(t))])

for i,ti in enumerate(t):
	etiq='t=%.2f'%ti
	plt.plot(x[::4],T[i][::4],label=etiq,marker=mk[i],markersize=9)
plt.legend()
plt.xlabel('$x$')
plt.ylabel(r'$T$[$^\circ$C]',rotation=90)
plt.savefig('/home/juan/Documents/Ensenanza/latex/tcmparcial/frente_trans.pdf')
plt.show()

plt.figure()

th=[x/np.sqrt(a*ti) for ti in t]

plt.plot(th,T)

