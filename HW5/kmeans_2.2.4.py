# This Python file uses the following encoding: utf-8
import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import random
import matplotlib.pyplot as plt
from random import randint
import math
import copy
#from ggplot import *
# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
pd.set_option('mode.chained_assignment', None) #avoid the warning
import warnings
warnings.simplefilter("ignore")
np.random.seed(0)
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1', header=None)
K =  4 # an integer to specify the number of clusters to use.
N = df.shape[0] #20000
num_iter = 50
#print df.shape,df.head(2)# (20000, 4)





######.............(1)............###########
X = df.values[0:,2:4]
label = df.values[0:,1:2].ravel()
#print type(label), label[0], type(label[0]), label
#print "class label", np.unique(label)
feat_one = df.values[0:,2:3]
feat_two = df.values[0:,3:4]
#print "feat_one", feat_one.ravel(), type(feat_one)
# Getting the values and plotting it
#feature = np.array(zip(feat_one, feat_two))
#print "feature", feature, len(feature), type(feature), feature[0], 'and',feature[0][0]
#plt.scatter(feat_one, feat_two, c='black', s=7)
#print "X", type(X[0]), X[0], X[0][0], len(X[0])

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# square Euclidean Distance Caculator
def sqr_dist(a, b, ax=1):
    return pow(np.linalg.norm(a - b, axis=ax),2)

def kmeans(X,label,N,num_iter,K):
	C_x = np.random.randint(0, np.max(X)-10, size=K)
	C_y = np.random.randint(0, np.max(X)-10, size=K)
	C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

	# To store the value of centroids when it updates
	C_old = np.zeros(C.shape)
	#print "C_old", C_old, type(C_old)

	# Cluster Lables(0, 1, 2, 3, 4)
	clusters = np.zeros(len(X))
	# Error func. - Distance between new centroids and old centroids
	error = dist(C, C_old, None)
	#print "dis", dist(X[0], C)
	#print "error", error
	# Loop will run till the error becomes zero
	epoch = 0
	points_clusterwise = []
	labels_cluster_arr =[]
	#num_empty_cluster = 0
	#while error != 0 or epoch < num_iter:
	while error != 0 and epoch < num_iter:
	    # Assigning each value to its closest cluster
	    #print "epoch", epoch
	    error = dist(C, C_old, None)
	    for i in range(len(X)):
	        distances = dist(X[i], C)
	        #print "loop distances", distances, len(distances), type(distances) #5
	        cluster = np.argmin(distances)
	        clusters[i] = cluster
	    #print "clusters", clusters
	    # Storing the old centroid values
	    C_old = copy.deepcopy(C)
	    # Finding the new centroids by taking the average value
	    for i in range(K):
	        points = [X[j] for j in range(len(X)) if clusters[j] == i]
	        labels_cluster = [label[j] for j in range(len(label)) if clusters[j] == i]

	        if len(points) == 0:
	        	#num_empty_cluster = num_empty_cluster +1
	        	p_x = np.random.randint(0, np.max(X)+100, size=K)
	        	p_y = np.random.randint(0, np.max(X)+100, size=K)
	        	points = np.array(list(zip(p_x, p_y)), dtype=np.float32)
	        	labels_cluster = np.array(np.random.randint(0, 9, size=K))

	        	#print "empty cluster"
	        #print "points", points, len(points), type(points)
	        #print "points_arr", points_arr, len(points_arr), type(points_arr)
	        temp = np.mean(points, axis=0)
	        C[i] = temp
	        if epoch == num_iter-1 or error == 0.0:
	    		points_clusterwise.append(points)
	    		labels_cluster_arr.append(labels_cluster)
	    		#break
	    epoch = epoch + 1
	    #error = dist(C, C_old, None)
	    #print "error",error



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
	return N_M_I, clusters, C
################# Dataset 1 ###############

N_M_I, clusters, C = kmeans(X, label, N, num_iter, K)
print "N_M_I for dataset 1:", round(N_M_I,2)
####plot#####
fig, ax = plt.subplots()
for i in range(K):
	points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
	if len(points) == 0:
		p_x = np.random.randint(0, np.max(X)+100, size=K)
		p_y = np.random.randint(0, np.max(X)+100, size=K)
		points = np.array(list(zip(p_x, p_y)), dtype=np.float32)

	#ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	ax.scatter(points[:, 0], points[:, 1])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
################### dataset 2 ###################

dataset2 = []
label2 = []
for i in range (0,N):
	if label[i] == 2 or label[i] == 4 or label[i] == 6 or label[i] == 7:
		dataset2.append(X[i])
		label2.append(label[i])
X_2 = np.asarray(dataset2)
Y_2 = np.asarray(label2)

N_M_I, clusters, C = kmeans(X_2, Y_2, N, num_iter, K)
print "N_M_I for dataset 2:", round(N_M_I,2)
####plot#####
fig, ax = plt.subplots()
for i in range(K):
	points = np.array([X_2[j] for j in range(len(X_2)) if clusters[j] == i])
	if len(points) == 0:
		p_x = np.random.randint(0, np.max(X_2)+100, size=K)
		p_y = np.random.randint(0, np.max(X_2)+100, size=K)
		points = np.array(list(zip(p_x, p_y)), dtype=np.float32)

	#ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	ax.scatter(points[:, 0], points[:, 1])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

################### dataset 3 ###################

dataset3 = []
label3 = []
for i in range (0,N):
	if label[i] == 6 or label[i] == 7:
		dataset3.append(X[i])
		label3.append(label[i])
X_3 = np.asarray(dataset3)
Y_3 = np.asarray(label3)
N_M_I, clusters, C = kmeans(X_3, Y_3, N, num_iter, K)
print "N_M_I for dataset 3:", round(N_M_I,2)
####plot#####
fig, ax = plt.subplots()
for i in range(K):
	points = np.array([X_3[j] for j in range(len(X_3)) if clusters[j] == i])
	if len(points) == 0:
		p_x = np.random.randint(0, np.max(X_3)+100, size=K)
		p_y = np.random.randint(0, np.max(X_3)+100, size=K)
		points = np.array(list(zip(p_x, p_y)), dtype=np.float32)

	#ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	ax.scatter(points[:, 0], points[:, 1])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

plt.show()

