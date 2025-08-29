import numpy as np
import datetime
import sys

print(sys.argv[1])

nombre_archivo = sys.argv[1]
A2 = open(nombre_archivo,'r')
milinea = A2.readline()
times,timed, timel, drag, lift = [[],[],[],[],[]]

while milinea:
    milinea = A2.readline()
    #print(milinea)
    if milinea!='':
        milinea1 = milinea.split('\t')
        milinea2 = milinea1[-1].split(':')
        if milinea2[-1] ==' \n':break
        if milinea2[0]=='Drag':
            drag.append(float(milinea2[-1].split('\n')[0]))
            time1 = datetime.datetime.strptime((milinea1[0]),'%H:%M:%S.%f')
            timed.append(time1.timestamp())
        if milinea2[0]=='Lift':
            lift.append(float(milinea2[-1].split('\n')[0]))         
            time1 = datetime.datetime.strptime((milinea1[0]),'%H:%M:%S.%f')
            timel.append(time1.timestamp())
        #print(milinea)
        time1 = datetime.datetime.strptime((milinea1[0]),'%H:%M:%S.%f')
        times.append(time1.timestamp())
A2.close()
times = np.asarray(times)
t0 = np.copy(times[0])
times = times - t0
timel = timel - t0
timed = timed - t0

out_drag = np.vstack((timed,drag))
out_lift = np.vstack((timel,lift))
nombre_drag = nombre_archivo.replace('.txt','_drag.txt')
nombre_lift = nombre_archivo.replace('.txt','_lift.txt')
np.savetxt(nombre_drag,out_drag,fmt='%.5f')
np.savetxt(nombre_lift,out_lift,fmt='%.5f')