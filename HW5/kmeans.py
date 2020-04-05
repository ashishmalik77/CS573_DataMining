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
K =  int(sys.argv[2]) # an integer to specify the number of clusters to use.
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
#print "np.max(X)-20",np.max(X)
# X coordinates of random centroids
#C_x = np.random.randint(0, np.max(X)-20, size=K)
C_x = np.random.randint(0, np.max(X)-10, size=K)
# Y coordinates of random centroids
#C_y = np.random.randint(0, np.max(X)-20, size=K)
C_y = np.random.randint(0, np.max(X)-10, size=K)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
#print "C",(C), len(C), type(C), C[0]

# Plotting along with the Centroids
#plt.scatter(feat_one, feat_two, c='#050505', s=7)
#plt.scatter(C_x, C_y, marker='*', s=200, c='g')

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

        if len(points) == 0: #handling empty cluster
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
'''
for i in range (0,len(labels_cluster_arr)):
	print "len",len(labels_cluster_arr[i])
	print "point len",len(points_clusterwise[i])
	#print "type",type(labels_cluster_arr[i]) #ok
	#print "pint type", type(points_clusterwise[i]) #ok
'''

#print "labels_cluster_arr", labels_cluster_arr
#print "len labels_cluster_arr",len(labels_cluster_arr) # K
#print "len labels_cluster_arr",len(labels_cluster_arr[0]), type(labels_cluster_arr[0])
#print "len labels_cluster_arr", type(labels_cluster_arr[0][0])
#print len(points_clusterwise)
#print len(points_clusterwise[1]),  len(points_clusterwise[0])
#print "clusters", clusters, type(clusters), len(clusters),clusters[0]
#print "num_empty_cluster", num_empty_cluster
#colors = ['r', 'g', 'b', 'y', 'c', 'm']
'''
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
plt.show()
'''
'''
points_clusterwise = []
labels_cluster_arr =[]

for i in range(K):
	#print i
	points = [X[j] for j in range(len(X)) if clusters[j] == i]
	if len(points) == 0: #handling empty cluster
		p_x = np.random.randint(0, np.max(X)+100, size=K)
		p_y = np.random.randint(0, np.max(X)+100, size=K)
		points = np.array(list(zip(p_x, p_y)), dtype=np.float32)

	#print points, type(points), len(points)
	labels_cluster = [label[j] for j in range(len(label)) if clusters[j] == i]

	points_clusterwise.append(points)
	labels_cluster_arr.append(labels_cluster)
	#temp = np.mean(points, axis=0)
'''
#print points, type(points), len(points)  #3651
#print "len points_clusterwise", len(points_clusterwise), len(points_clusterwise[0]), len(points_clusterwise[0][0])  # 5 5235 2
#print "type points_clusterwise", type(points_clusterwise), type(points_clusterwise[0]), type(points_clusterwise[0][0])  
#print " points_clusterwise", len(points_clusterwise),points_clusterwise,points_clusterwise[1],len(points_clusterwise[1]),points_clusterwise[1][0]
#print " points_clusterwise", len(points_clusterwise[0]),len(points_clusterwise[1])
#print "labels_cluster", labels_cluster_arr, len(labels_cluster_arr),len(labels_cluster_arr[0]),len(labels_cluster_arr[1]),len(labels_cluster_arr[2]),len(labels_cluster_arr[3]),len(labels_cluster_arr[4])
#print "labels_cluster_arr[0]",labels_cluster_arr[0]

########### Calculate within cluster sum of squared distances ############

WC_SSD = []
#sum_squared = 0
for i in range(K):	
	#distance_intra_cluster = []
	sum_squared = 0
	#print len(points_clusterwise[i])
	for j in range (0,len(points_clusterwise[i])):
		#dist(X[i], C)
		disrance_A = sqr_dist(points_clusterwise[i][j], points_clusterwise[i])
		sum_squared = sum_squared + disrance_A
		#print "disrance_A", disrance_A, type (disrance_A), len(disrance_A) #5235
		#distance_intra_cluster.append(np.sum(disrance_A))
		#distance_intra_cluster.append((disrance_A))
	#print "distance_intra_cluster", len(distance_intra_cluster)
	#WC_SSD.append(np.sum(distance_intra_cluster))
	WC_SSD.append(np.mean(sum_squared))
#print "WC-SSD: ", len(WC_SSD),len(WC_SSD[0]), len(WC_SSD[0][0]), type(WC_SSD[0][0]) # 5 5235 5235 <type 'numpy.ndarray'>
#print len(WC_SSD), len(WC_SSD[0])
#print "distance_intra_cluster: ",distance_intra_cluster,len(distance_intra_cluster),  (distance_intra_cluster[0]), len(distance_intra_cluster[0][0])
print "WC-SSD: ", round(np.sum(WC_SSD),2)
#print "WC-SSD: ", round(np.mean(WC_SSD),2)
############## Calculate Silhoutte Co-efficient ###################  
A = []
for i in range(K):
	distance_intra_cluster = []
	for j in range (0,len(points_clusterwise[i])):
		#dist(X[i], C)
		disrance_A = dist(points_clusterwise[i][j], points_clusterwise[i])
		distance_intra_cluster.append(disrance_A)
	#print "distance_intra_cluster", len(distance_intra_cluster)
	A.append(np.mean(distance_intra_cluster)) 
#print A, len(A) #5

B = []
for i in range(K):
	#print i
	if i == K-1:
		B.append(np.mean(B)) 
	else:
		distance_inter_cluster = []
		p = i + 1
		while p < K :
			#print "p", p
			for j in range (0,len(points_clusterwise[p])):
				#dist(X[i], C)
				disrance_B = dist(points_clusterwise[p][j], points_clusterwise[i])
				distance_inter_cluster.append(disrance_B)
			p = p + 1
		B.append(np.mean(distance_inter_cluster)) 
#print B, len(B) #10

SC = []
for i in range (0,len(A)):
	#print np.maximum(A[i],B[i])
	SC_i = float(B[i]-A[i]) / np.maximum(A[i],B[i])
	SC.append(SC_i)
print "SC: ", round(np.mean(SC),2)

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
#print "denominator", denominator

#print "labels_cluster_arr[0]", labels_cluster_arr[0]
#print "labels_cluster_arr",labels_cluster_arr[0][0],labels_cluster_arr[0][1],labels_cluster_arr[0][3], label[0], label[1]

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
#print "p_c_g", p_c_g, len(p_c_g)
#print "labels_cluster_arr", len(labels_cluster_arr), labels_cluster_arr[0]
#print "count1 ", count1 
'''
numerator  = 0
for i in range(0, len(p_c)):
	#numerator = 0
	for j in range (0,len(p_c_g)):
		#numerator = numerator + (p_c_g[j] * (np.log(p_c_g[j]) - np.log(p_g[j] * p_c[i])))
		numerator = numerator + (p_c_g[j] * (np.log(float(p_c_g[j]) / ( p_g[j] * p_c[i]))))

'''
numerator_arr = []
for i in range (0,len(p_c)):
	numerator = 0
	for j in range (0,len(p_c_g)):
		#numerator = numerator + (p_c_g[j] * (np.log(p_c_g[j]) - np.log(p_g[j] * p_c[i])))
		numerator = numerator + (p_c_g[j] * (np.log(float(p_c_g[j]) / ( p_g[j] * p_c[i] ))))
	numerator_arr.append(numerator)

#print "numerator mean", (numerator_arr)

N_M_I = float(np.mean(numerator_arr))/denominator
print "N_M_I :", round(N_M_I,2)



