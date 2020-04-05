# This Python file uses the following encoding: utf-8
import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import random
import matplotlib.pyplot as plt
from random import randint
import matplotlib
#from ggplot import *
# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
pd.set_option('mode.chained_assignment', None) #avoid the warning
df1 = pd.read_csv( sys.argv[1], encoding='ISO-8859-1', header=None)
df2 = pd.read_csv( sys.argv[2], encoding='ISO-8859-1', header=None)

#print df1.shape,df1.head(2)# (19999, 786)
#print df2.shape #(19999, 4)




######.............(1)............###########
label = df1.values[0:,1:2]
feature = df1.values[0:,2:786]

#print label, type(label)
#print np.unique(label)
#print X[0]
#print y[0]

#X = pd.DataFrame(feature)
#y = pd.DataFrame(label)

np.random.seed(0)
picked_digit = np.random.randint(0, 9)
#print picked_digit

picked_index = 0
for i in range (0,df1.shape[0]):
	if picked_digit == label[i]:
		#print label[i]
		picked_index = i
		break
#print picked_index

# First row is first image
#first_image = X.loc[picked_index,:]
#first_label = y[picked_index]

# 784 columns correspond to 28x28 image
#plottable_image = np.reshape(first_image.values, (28, 28))
plottable_image = np.reshape(feature[picked_index,:], (28, 28))

# Plot the image
plt.imshow(plottable_image, cmap='gray_r')
plt.title('Digit Label: {}'.format(label[picked_index,:]))

plt.show()
#####............(2).................########
N = df2.shape[0]
examples =  np.random.randint(0, N, size=1000)
#print examples #ok
x_axis = []
y_axis = []
class_label = []
#print df2.loc[2,1]
for i in examples:
	class_label.append(df2.loc[i,1])
	x_axis.append(df2.values[i,2])
	y_axis.append(df2.values[i,3])
	#print i

#print class_label
#print len( class_label)
label = np.unique(class_label)
#print label
#print len( label)

'''

import pylab
colors = [int(i) for i in class_label]
pylab.scatter(x_axis, y_axis, c=colors, cmap=pylab.cm.cool)
pylab.show()
'''



#x = [4,8,12,16,1,4,9,16]
#y = [1,4,9,16,4,8,12,3]
#label = [0,1,2,3,0,1,2,3]
import pylab
colors = ['red','green','blue','purple','gray','brown','cyan','orange','magenta']

pylab.scatter(x_axis, y_axis, c=class_label, cmap=matplotlib.colors.ListedColormap(colors))

cb = pylab.colorbar()
loc = np.arange(0,max(class_label),max(class_label)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)

pylab.show()




#np.random.seed(0)
