import numpy as np 
import matplotlib.pyplot as plt
import pylab
from scipy import optimize
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'text.fontsize': 16,
          'legend.fontsize': 18,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'text.usetex': True,
          'figure.figsize': (10,8)}
pylab.rcParams.update(params)
from scipy import optimize

datat=open('tablav.txt','r')
dd=datat.readlines()
T=np.asarray([np.float(dd[i].split('\t')[0]) for i in range(len(dd))])
pvs=np.asarray([np.float(dd[i].split('\t')[1]) for i in range(len(dd))])*101.3/760.
rho=np.asarray([np.float(dd[i].split('\t')[2][:-1]) for i in range(len(dd))])
#plt.plot(T,pvs*1014./760.,'b-',linewidth=4)
plt.plot(T,pvs,'ro',markersize=10)

plt.axis([-10,120,.1,120]) ; plt.grid()
plt.ylabel('$p_{vs}$(kPa)') ; plt.xlabel('$T$($^\circ$C)')

fitfunc = lambda p, x: p[0]+p[1]*np.exp(p[2]*x)+x*p[3]*np.exp(p[4]*x**2)
#fitfunc2 = lambda p, x: p[0]+p[1]*x+ p[2]*x**2+ p[3]*x**3. 
errfunc = lambda p, x, y: fitfunc(p, x) - y
#errfunc2 = lambda p, x, y: fitfunc(p, x) - y
p0 = [110., 0., 0., 0.,0.,0.,0.,0.,0,0] # Initial guess for the parameters

p1, success = optimize.leastsq(errfunc, p0[:], args=(T[:-3],pvs[:-3]))
Tx=np.linspace(T[0],T[-3],100)

plt.semilogy(Tx,fitfunc(p1,Tx),color='blue',linewidth=4,linestyle='--')
plt.show()
#for i in range(nn):
                         #tmp=coordx.readline()
                         #if len(tmp)>3:
                                 #xi[i]=map(float,tmp[9:-3].split(' '))[0]
