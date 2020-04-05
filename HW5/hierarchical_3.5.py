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
from scipy.cluster.hierarchy import dendrogram, linkage, centroid, fcluster

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
label_new = []
Y_100 =[]
for i in examples_1d:
	#class_label.append(df2.loc[i,1])
	x_axis.append(df.values[i,2])
	y_axis.append(df.values[i,3])
	label_new.append(df.values[i,1])
	#X_100.append(df.values[i,2:4])
	#print i
#print "x_axis", len(x_axis), x_axis
#print "y_axis", len(y_axis), y_axis
X_100 = np.array(list(zip(x_axis, y_axis)), dtype=np.float32)
#print "X_100", X_100, type(X_100), len(X_100)
#print "label_new", label_new
#Y_100 = np.asarray(label_new)
#print "Y_100", Y_100

num_iter = 50
N = len(X_100)
K = 4

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# square Euclidean Distance Caculator
def sqr_dist(a, b, ax=1):
    return pow(np.linalg.norm(a - b, axis=ax),2)
def hierarchical(X,label,N,num_iter,K,clusters):
	epoch = 0
	points_clusterwise = []
	labels_cluster_arr =[]
	#print "len ", len(X), len(clusters)
	while epoch < num_iter:
	    for i in range(K):
	        points = [X[j] for j in range(len(X)) if clusters[j] == i]
	        labels_cluster = [label[j] for j in range(len(label)) if clusters[j] == i]

        	if len(points) == 0: #handling empty cluster
	        	#num_empty_cluster = num_empty_cluster +1
	        	p_x = np.random.randint(0, np.max(X)+100, size=K)
	        	p_y = np.random.randint(0, np.max(X)+100, size=K)
	        	points = np.array(list(zip(p_x, p_y)), dtype=np.float32)
	        	labels_cluster = np.array(np.random.randint(0, 9, size=K))
	        if epoch == num_iter-1 :
    			points_clusterwise.append(points)
    			labels_cluster_arr.append(labels_cluster)
	    epoch = epoch + 1

	############## Calculate NMI #############

	####### Entropy of Class Labels #########
	p_c = []
	class_label = np.unique(label)
	#print class_label, type(class_label), class_label[0], type(class_label[0])
	for i in range(0, len(class_label)):
		count = 0
		for j in range (0,len(label)):	
			if label[j] == class_label[i]:
				#print "hellp"
				count = count + 1
		p_c.append(float(count)/N)

	#print p_c

	denom_1st_arr = []
	for i in range (0,len(p_c)):
		denom_1st_arr.append(p_c[i] * np.log(p_c[i]))
	#print "denom_1st_arr", denom_1st_arr
	denom_1st = sum(denom_1st_arr)
	#print "denom_1st",denom_1st



	####### Entropy of Cluster Labels #########
	p_g = []
	for i in range(K):	
		count = 0
		for j in range (0,len(points_clusterwise[i])):
			count = count + 1
		p_g.append(float(count)/N)
	#print p_g

	denom_2nd_arr = []
	for i in range (0,len(p_g)):
		denom_2nd_arr.append(p_g[i] * np.log(p_g[i]))
	#print "denom_1st_arr", denom_1st_arr
	denom_2nd = sum(denom_2nd_arr)
	#print "denom_2nd",denom_2nd


	denominator = - denom_1st - denom_2nd

	########## calculate Mutual Information ###########
	p_c_g = []
	for i in range(K):	
		count1 = 0
		for p in range (0,len(class_label)):
		#for j in range (0,len(labels_cluster_arr[i])):
			#count1 = 0
			#for p in range (0,len(class_label)):
			for j in range (0,len(labels_cluster_arr[i])):
				#print "X[p]", X[p], type(X[p])
				#print "points_clusterwise[i][j]", points_clusterwise[i][j],type(points_clusterwise[i][j])
				if labels_cluster_arr[i][j] == class_label[p]:
					count1 = count1 + 1
					#labels_cluster_arr.append(label[p])
		p_c_g.append(float(count1)/N)

	numerator_arr = []
	for i in range (0,len(p_c)):
		numerator = 0
		for j in range (0,len(p_c_g)):
			#numerator = numerator + (p_c_g[j] * (np.log(p_c_g[j]) - np.log(p_g[j] * p_c[i])))
			numerator = numerator + (p_c_g[j] * (np.log(float(p_c_g[j]) / ( p_g[j] * p_c[i] ))))
		numerator_arr.append(numerator)
	N_M_I = float(np.mean(numerator_arr))/denominator
	return N_M_I

##########...........Single.........##########
linked_sing = linkage(X_100, 'single')
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList

T3 = fcluster(linked_sing, K, criterion='maxclust' )
N_M_I = hierarchical(X_100,label_new,N,num_iter,K, T3)
print "N_M_I for Single Linkage:", round(N_M_I,2)


######...........Complete..............############


linked_com = linkage(X_100, 'complete')
T3 = fcluster(linked_com, K, criterion='maxclust' )
N_M_I = hierarchical(X_100,label_new,N,num_iter,K, T3)
print "N_M_I for Complete Linkage:", round(N_M_I,2)

#########........Average............###########
linked_avg = linkage(X_100, 'average')
T3 = fcluster(linked_avg, K, criterion='maxclust' )
N_M_I = hierarchical(X_100,label_new,N,num_iter,K, T3)
print "N_M_I for Average Linkage:", round(N_M_I,2)

