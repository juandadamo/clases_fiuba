
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt


# In[1]:

import numpy as np
import matplotlib.pyplot as plt

#initializing variables
g = 70
# u --> i --- >x 
u = np.zeros((g+1, g))
un = np.zeros((g+1, g))
# v --> j ||| >z 
v = np.zeros((g, g+1))
vn = np.zeros((g, g+1))

m = np.ones((g+1, g+1))
mn = np.ones((g+1, g+1))

#variables: 
temp = 10.0              #temperature
steps = 1500
t = 1500.0
dt = t/float(steps)
dx = 100.0/float(g)
iter=0
#setting the initial condition of m
for i in range(g+1):
    for j in range(g+1):
        if((i>30 and i<60 and j>1 and j<10)or (i>2 and i<10 and j>40 and j<50)):
            m[i][j] = temp
#print m
for x in range(steps):
    t = t+dt
    for i in range(1, g-1):
        for j in range(1, g-1):
            mn[i][j] = m[i][j] +            ((0.02*dt/dx**2)*(m[i+1][j]- 2* m[i][j]+ m[i-1][j]))+            ((0.02*dt/dx**2)*(m[i][j+1]- 2* m[i][j]+ m[i][j-1])) + 0.005
    for i in range(g):
        for j in range(g):
            m[i][j] = mn[i][j]
    if(x%5 == 0):
        f, ((plt1, plt2)) = plt.subplots(1, 2, figsize=(8.5, 4.0))
        plt.suptitle("step" + str(x))
        f = plt1.pcolor(np.transpose(mn), cmap="hot")
        f = plt2.contourf(np.transpose(mn))
        plt.savefig("E:\\g\\" + str(iter)+".png")
        print str(iter)+ " saved at E:\\g\\" + str(iter)+".png"
        iter+=1


# In[39]:




# In[ ]:



