
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

mk=np.tile(['o','s','<','D','>','h','p','x','d','^'],3)
ck=np.tile(['k','r','b','g','m','y','c','k'],3)
pylab.rcParams.update(params)
plt.close('all')


sig=5.67e-8

la1=np.linspace(-10,1,1000)

la2=10**(la1)
C1=3.742*1e8*(1e-6)**4
C2=14388*1e-6
T=[5400.,2000.,1000.,500.,300.]
for k,Ti in enumerate(T):
	Ebl=C1*la2**(-5)/ ( np.exp(C2/la2/Ti)-1)
	etiq='$T=%0d~K$'%Ti
	plt.semilogx((la2*1e6),np.log10(Ebl), color=ck[k],linewidth=3,label=etiq)
	#raise ValueError('alto')
plt.xlim([0.03,5000])
plt.ylim([0.01,15])

lm=np.asarray([2897.8/Ti for Ti in T])*1e-6

Eblm=C1*lm**(-5)/ ( np.exp(C2/lm/T)-1)
k=0
for k,lmi in enumerate(lm):    
    plt.semilogx((lmi*1e6),np.log10(Eblm[k]),marker='s',linestyle='--',color=ck[k],markersize=10)
plt.semilogx((lm*1e6),np.log10(Eblm),linestyle='--',color='k',label='Ley de Wien',linewidth=2)    
plt.xlabel(r'$\lambda[\mu m]$')
plt.ylabel(r'$E_{b\lambda}$')
plt.legend(numpoints=1)	
plt.tight_layout()
plt.savefig('/home/juan/Documents/Ensenanza/latex/radiacion/planck.pdf')
#Eblm=[]
#for i,lmi in enumerate(lm):
	#Ti=T[i]
	#Eblm.append(C1*lmi**(-5)/ ( np.exp(C2/lmi/Ti)-1) )


#for k,Ti in enumerate(T):
	#plt.loglog(la1,Ebl[k],label=str(Ti))
#plt.axis([0.0001,25,0.1,1e25])
#plt.legend()
#plt.show()

##al=1.
##la1=50.
##la2=.08
##L=10.
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