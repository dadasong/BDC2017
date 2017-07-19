
# coding: utf-8

# In[1]:

import numpy as np 
import pandas as pd
import string
import os
import re
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from scipy import stats

#读
training_list = open("C:/Users/Administrator/Desktop/mouse path/dsjtzs_txfz_training.txt")
#test_sample = open("dsjtzs_txfz_test_sample.txt")

training_list = training_list.readlines()

for j in range(5):
    time = []
    x = []
    y = []
    x_cordinate = []
    y_cordinate = []
    time_cordinate = []
    for i in training_list[j][0:-1].split(" ")[1].split(";")[0:-1]:
        x.append(i.split(",")[0])
        y.append(i.split(",")[1])
        time.append(i.split(",")[2])

    for i in range(len(x)):
        x_cordinate.append(float(x[i]))
    
    for i in range(len(y)):
        y_cordinate.append(float(y[i]))
    
    for i in range(len(time)):
        time_cordinate.append(float(time[i]))



# In[8]:

fig = plt.figure()
ax1 = fig.add_subplot(211)
x = y_cordinate
prob = stats.probplot(x, dist = stats.norm, plot = ax1)

ax2 = fig.add_subplot(212)
xt, _ = stats.boxcox(x)
prob = stats.probplot(xt, dist=stats.norm, plot=ax2,)
plt.show()


# In[13]:

x


# In[ ]:



