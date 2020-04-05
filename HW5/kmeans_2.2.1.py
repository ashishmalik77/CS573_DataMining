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
#K =  int(sys.argv[2]) # an integer to specify the number of clusters to use.
K = [2, 4, 8, 16, 32]
#K = [32]
N = df.shape[0] #20000
num_iter = 50
#print df.shape,df.head(2)# (20000, 4)




######.............dataset 1............###########
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

	# X coordinates of random centroids
	C_x = np.random.randint(0, np.max(X)-10, size=K)
	# Y coordinates of random centroids
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
	while error != 0 and epoch < num_iter:
	#while error != 0:
	    # Assigning each value to its closest cluster
	    error = dist(C, C_old, None)
	    for i in range(len(X)):
	        distances = dist(X[i], C)
	        #print "loop distances", distances, len(distances), type(distances) #5
	        cluster = np.argmin(distances)
	        clusters[i] = cluster
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
	        #print "points", points, len(points), type(points)
	        #print "points_arr", points_arr, len(points_arr), type(points_arr)
	        temp = np.mean(points, axis=0)
	        C[i] = temp
	        if epoch == num_iter-1 or error == 0.0:
    			points_clusterwise.append(points)
    			labels_cluster_arr.append(labels_cluster)
	    epoch = epoch + 1
	    #error = dist(C, C_old, None)
	#print "clusters", clusters, type(clusters), len(clusters),clusters[0]
	'''
	points_clusterwise = []
	labels_cluster_arr =[]
	for i in range(K[m]):
		#print i
		points = [X[j] for j in range(len(X)) if clusters[j] == i]
		labels_cluster = [label[j] for j in range(len(label)) if clusters[j] == i]
		points_clusterwise.append(points)
		labels_cluster_arr.append(labels_cluster)
	#print "points_clusterwise", type (points_clusterwise)
	#labels_cluster_arr_K.append(labels_cluster_arr)
	'''
	########### Calculate within cluster sum of squared distances ############

	WC_SSD = []

	for i in range(K):	
		#distance_intra_cluster = []
		sum_squared = 0
		for j in range (0,len(points_clusterwise[i])):
			#dist(X[i], C)
			disrance_A = sqr_dist(np.array(points_clusterwise[i][j]), np.array(points_clusterwise[i]))
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
	#print "WC-SSD: ", round(np.sum(WC_SSD),2)
	#WC_SSD_K.append(np.sum(WC_SSD))
	
	############## Calculate Silhoutte Co-efficient ###################  
	A = []
	for i in range(K):
		distance_intra_cluster = []
		for j in range (0,len(points_clusterwise[i])):
			#dist(X[i], C)
			disrance_A = dist(np.array(points_clusterwise[i][j]), np.array(points_clusterwise[i]))
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
					disrance_B = dist(np.array(points_clusterwise[p][j]), np.array(points_clusterwise[i]))
					distance_inter_cluster.append(disrance_B)
				p = p + 1
			B.append(np.mean(distance_inter_cluster)) 
	#print B, len(B) #10

	SC = []
	for i in range (0,len(A)):
		#print np.maximum(A[i],B[i])
		SC_i = float(B[i]-A[i]) / np.maximum(A[i],B[i])
		SC.append(SC_i)
	#print "SC: ", round(np.mean(SC),2)
	#SC_K.append(np.mean(SC))
	return WC_SSD, SC

#print "WC_SSD_K", WC_SSD_K

#print " SC_K", SC_K

WC_SSD_ds1 = []
SC_ds1 =[]
for m in range (0,len(K)):
	WC_SSD, SC = kmeans(X,label,N,num_iter,K[m])
	WC_SSD_ds1.append(np.mean(WC_SSD))
	SC_ds1.append(np.mean(SC))


n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)

plt.plot( WC_SSD_ds1, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
#plt.plot( SC_ds1, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('WC-SSD as a function of K for dataset1')
plt.xticks(index, ('2', '4', '8', '16', '32'))
plt.legend()
plt.tight_layout()


n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)

#plt.plot( WC_SSD_ds1, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
plt.plot( SC_ds1, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('SC as a function of K for dataset1')
plt.xticks(index, ('2', '4', '8', '16', '32'))
plt.legend()
plt.tight_layout()


######.......Dataset 2 (data consisting of the digits 2, 4, 6 and 7).......... ########

dataset2 = []
label2 = []
for i in range (0,N):
	if label[i] == 2 or label[i] == 4 or label[i] == 6 or label[i] == 7:
		dataset2.append(X[i])
		label2.append(label[i])
X_2 = np.asarray(dataset2)
Y_2 = np.asarray(label2)
WC_SSD_ds2 = []
SC_ds2 =[]
for m in range (0,len(K)):
	WC_SSD, SC = kmeans(X_2,Y_2,N,num_iter,K[m])
	WC_SSD_ds2.append(np.mean(WC_SSD))
	SC_ds2.append(np.mean(SC))

n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)

plt.plot( WC_SSD_ds2, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
#plt.plot( SC_ds2, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('WC-SSD as a function of K for dataset2')
plt.xticks(index, ('2', '4', '8', '16', '32'))
plt.legend()
plt.tight_layout()

n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)
#plt.plot( WC_SSD_ds2, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
plt.plot( SC_ds2, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('SC as a function of K for dataset2')
plt.xticks(index, ('2', '4', '8', '16', '32'))
plt.legend()
plt.tight_layout()

######.......Dataset 3 (data consisting of the digits 6 and 7).......... ########

dataset3 = []
label3 = []
for i in range (0,N):
	if label[i] == 6 or label[i] == 7:
		dataset3.append(X[i])
		label3.append(label[i])
X_3 = np.asarray(dataset3)
Y_3 = np.asarray(label3)
#print len(X_3), type(X_3), type(X)
WC_SSD_ds3 = []
SC_ds3 =[]
for m in range (0,len(K)):
	WC_SSD, SC = kmeans(X_3,Y_3,N,num_iter,K[m])
	WC_SSD_ds3.append(np.mean(WC_SSD))
	SC_ds3.append(np.mean(SC))

n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)

plt.plot( WC_SSD_ds3, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
#plt.plot( SC_ds3, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('WC-SSD as a function of K for dataset3')
plt.xticks(index, ('2', '4', '8', '16', '32'))
plt.legend()
plt.tight_layout()


n_groups = 5
fig, ax = plt.subplots()
index = np.arange(n_groups)

#plt.plot( WC_SSD_ds3, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
plt.plot( SC_ds3, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('SC as a function of K for dataset3')
plt.xticks(index, ('2', '4', '8', '16', '32'))
plt.legend()
plt.tight_layout()

plt.show()


