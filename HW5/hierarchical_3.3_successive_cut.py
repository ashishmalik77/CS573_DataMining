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
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage 
#from ggplot import *
# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
pd.set_option('mode.chained_assignment', None) #avoid the warning
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1', header=None)
np.random.seed(0)
#####............(2).................########
N = df.shape[0]

X = df.values[0:,2:4]
label = df.values[0:,1:2].ravel()

class_label = np.unique(label)

examples = []
for i in range (0,len(class_label)):
	X_new = []
	for j in range (0,len(label)):
		if class_label[i] == label[j]:
			X_new.append(X[j]) 
	examples.append(np.random.randint(0, len(X_new), size=10))


#examples =  np.random.randint(0, N, size=10)
#print "examples",examples, len(examples), examples[0], len(examples[0])  #ok 10
examples_1d = (np.asarray(examples)).ravel()
#print "examples_1d", examples_1d, len(examples_1d) #ok 100
x_axis = []
y_axis = []
#class_label = []
#print df2.loc[2,1]
X_100 = []
for i in examples_1d:
	#class_label.append(df2.loc[i,1])
	x_axis.append(df.values[i,2])
	y_axis.append(df.values[i,3])
	#X_100.append(df.values[i,2:4])
	#print i
#print "x_axis", len(x_axis), x_axis
#print "y_axis", len(y_axis), y_axis
X_100 = np.array(list(zip(x_axis, y_axis)), dtype=np.float32)
#print "X_100", X_100, type(X_100), len(X_100)

'''
labels = range(1, 11)  
plt.figure(figsize=(10, 7))  
plt.subplots_adjust(bottom=0.1)  
plt.scatter(x_axis,y_axis, label='True Position')

for label, x, y in zip(labels,x_axis, y_axis):  
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show() 

'''

linked_sing = linkage(X_100, 'single')
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList
plt.figure(figsize=(10, 7))  
plt.axhline(y=10, xmin=0, xmax=1, linestyle = '--' ,hold=None, label='K=4') # K = 4
plt.axhline(y=8.5, xmin=0, xmax=1, linestyle = ':' ,hold=None, label='K=8') # K = 8
dendrogram(linked_sing)
plt.legend()
linked_com = linkage(X_100, 'complete')
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList
plt.figure(figsize=(10, 7))  
plt.axhline(y=55, xmin=0, xmax=1, linestyle = '--' ,hold=None, label='K=4') # K = 4
plt.axhline(y=31, xmin=0, xmax=1, linestyle = ':' ,hold=None, label='K=8') # K = 8
dendrogram(linked_com)
plt.legend()


linked_avg = linkage(X_100, 'average')
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList
plt.figure(figsize=(10, 7))
plt.axhline(y=30, xmin=0, xmax=1, linestyle = '--' ,hold=None, label='K=4') # K = 4  
plt.axhline(y=20, xmin=0, xmax=1, linestyle = ':' ,hold=None, label='K=8') # K = 8  
dendrogram(linked_avg)
plt.legend()
plt.show() 
'''
linked = linkage(X_100, 'single')
P = dendrogram(linked)
pos = None
plt.clf()
icoord = scipy.array(P['icoord'])
#print icoord
dcoord = scipy.array(P['dcoord'])
color_list = scipy.array(P['color_list'])
xmin, xmax = icoord.min(), icoord.max()
ymin, ymax = dcoord.min(), dcoord.max()
if pos:
    icoord = icoord[pos]
    dcoord = dcoord[pos]
    color_list = color_list[pos]
for xs, ys, color in zip(icoord, dcoord, color_list):
    plt.plot(xs, ys, color)
plt.xlim(xmin-10, xmax + 0.1*abs(xmax))
plt.ylim(ymin, ymax + 0.1*abs(ymax))
plt.show()
'''