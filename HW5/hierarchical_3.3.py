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


#K = [2,4,6,8]


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

##########...........Single.........##########
linked_sing = linkage(X_100, 'single')
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList

T1 = fcluster(linked_sing, 8, criterion='maxclust' )
#print "T1", T1, len(T1), type(T1), np.unique(T1)
T2 = fcluster(linked_sing, 6, criterion='maxclust' )
#print "T2", T2, len(T2), type(T2), np.unique(T2)
T3 = fcluster(linked_sing, 4, criterion='maxclust' )
#print "T3", T3, len(T3), type(T3), np.unique(T3)
T4 = fcluster(linked_sing, 2, criterion='maxclust' )
#print "T4", T4, len(T4), type(T4), np.unique(T4)

#from sklearn.metrics import silhouette_score
#print silhouette_score(X_100 , T3, metric='euclidean') # 0.11532326 K = 4

'''
T1 = fcluster(linked_sing, 8, criterion='distance' )
print "T1", T1, len(T1), type(T1), np.unique(T1)
T2 = fcluster(linked_sing, 6, criterion='distance' )
print "T2", T2, len(T2), type(T2), np.unique(T2)
T3 = fcluster(linked_sing, 4, criterion='distance' )
print "T3", T3, len(T3), type(T3), np.unique(T3)
T4 = fcluster(linked_sing, 2, criterion='distance' )
print "T4", T4, len(T4), type(T4), np.unique(T4)
'''
#plt.figure(figsize=(10, 7)) 
#dendrogram(ddgm) 
#dendrogram(linked_sing)
#plt.axhline(y=10, xmin=0, xmax=1, linestyle = '--' ,hold=None) # K = 4

#C = centroid(linked_sing)
#print "centroid", C, type(C), len(C)
num_iter = 50
N = len(X_100)

K = 2
WC_SSD2, SC2 = hierarchical(X_100,label_new,N,num_iter,K, T4)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD=  78940.14
#print "SC", np.mean(SC) #SC= 0.44766595437047507

K = 4
WC_SSD4, SC4 = hierarchical(X_100,label_new,N,num_iter,K, T3)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 30732.52
#print "SC", np.mean(SC) #SC= 0.6800625177957564

K = 6
WC_SSD6, SC6 = hierarchical(X_100,label_new,N,num_iter,K, T2)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 20248.533
#print "SC", np.mean(SC) #SC= 0.6

K = 8
WC_SSD8, SC8 = hierarchical(X_100,label_new,N,num_iter,K, T1)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 11616.799
#print "SC", np.mean(SC) #SC= 0.68808
WC_SSD_arr_single = [np.mean(WC_SSD2), np.mean(WC_SSD4), np.mean(WC_SSD6), np.mean(WC_SSD8)]
#print type(WC_SSD_arr_single)
SC_arr_single = [np.mean(SC2), np.mean(SC4), np.mean(SC6), np.mean(SC8)]


n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)

plt.plot( WC_SSD_arr_single, '-bo', label='within-cluster sum of squared distances (WC-SSD)') 
#plt.plot( SC_arr_single, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('WC-SSD as a function of K for Single linkage')
plt.xticks(index, ('2', '4', '6', '8'))
plt.legend()
plt.tight_layout()


n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)

#plt.plot( WC_SSD_arr_single, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
plt.plot( SC_arr_single, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('SC as a function of K for Single linkage')
plt.xticks(index, ('2', '4', '6', '8'))
plt.legend()
plt.tight_layout()

######...........Complete..............############


linked_com = linkage(X_100, 'complete')
T1 = fcluster(linked_com, 8, criterion='maxclust' )
#print "T1", T1, len(T1), type(T1), np.unique(T1)
T2 = fcluster(linked_com, 6, criterion='maxclust' )
#print "T2", T2, len(T2), type(T2), np.unique(T2)
T3 = fcluster(linked_com, 4, criterion='maxclust' )
#print "T3", T3, len(T3), type(T3), np.unique(T3)
T4 = fcluster(linked_com, 2, criterion='maxclust' )
#print "T4", T4, len(T4), type(T4), np.unique(T4)

K = 2
WC_SSD2, SC2 = hierarchical(X_100,label_new,N,num_iter,K, T4)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD=  78940.14
#print "SC", np.mean(SC) #SC= 0.44766595437047507

K = 4
WC_SSD4, SC4 = hierarchical(X_100,label_new,N,num_iter,K, T3)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 30732.52
#print "SC", np.mean(SC) #SC= 0.6800625177957564

K = 6
WC_SSD6, SC6 = hierarchical(X_100,label_new,N,num_iter,K, T2)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 20248.533
#print "SC", np.mean(SC) #SC= 0.6

K = 8
WC_SSD8, SC8 = hierarchical(X_100,label_new,N,num_iter,K, T1)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 11616.799
#print "SC", np.mean(SC) #SC= 0.68808
WC_SSD_arr_complete = [np.mean(WC_SSD2), np.mean(WC_SSD4), np.mean(WC_SSD6), np.mean(WC_SSD8)]
SC_arr_complete = [np.mean(SC2), np.mean(SC4), np.mean(SC6), np.mean(SC8)]
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList
#plt.figure(figsize=(10, 7))  
#dendrogram(linked_com)

n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)

plt.plot( WC_SSD_arr_complete, '-bo', label='within-cluster sum of squared distances (WC-SSD)') 
#plt.plot( SC_arr_complete, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('WC-SSD as a function of K for Complete linkage')
plt.xticks(index, ('2', '4', '6', '8'))
plt.legend()
plt.tight_layout()


n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)

#plt.plot( WC_SSD_arr_complete, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
plt.plot( SC_arr_complete, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('SC as a function of K for Complete linkage')
plt.xticks(index, ('2', '4', '6', '8'))
plt.legend()
plt.tight_layout()

#########........Average............###########
linked_avg = linkage(X_100, 'average')
T1 = fcluster(linked_avg, 8, criterion='maxclust' )
#print "T1", T1, len(T1), type(T1), np.unique(T1)
T2 = fcluster(linked_avg, 6, criterion='maxclust' )
#print "T2", T2, len(T2), type(T2), np.unique(T2)
T3 = fcluster(linked_avg, 4, criterion='maxclust' )
#print "T3", T3, len(T3), type(T3), np.unique(T3)
T4 = fcluster(linked_avg, 2, criterion='maxclust' )
#print "T4", T4, len(T4), type(T4), np.unique(T4)

K = 2
WC_SSD2, SC2 = hierarchical(X_100,label_new,N,num_iter,K, T4)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD=  78940.14
#print "SC", np.mean(SC) #SC= 0.44766595437047507

K = 4
WC_SSD4, SC4 = hierarchical(X_100,label_new,N,num_iter,K, T3)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 30732.52
#print "SC", np.mean(SC) #SC= 0.6800625177957564

K = 6
WC_SSD6, SC6 = hierarchical(X_100,label_new,N,num_iter,K, T2)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 20248.533
#print "SC", np.mean(SC) #SC= 0.6

K = 8
WC_SSD8, SC8 = hierarchical(X_100,label_new,N,num_iter,K, T1)
#print "WC_SSD", np.mean(WC_SSD) #WC_SSD= 11616.799
#print "SC", np.mean(SC) #SC= 0.68808
WC_SSD_arr_average = [np.mean(WC_SSD2), np.mean(WC_SSD4), np.mean(WC_SSD6), np.mean(WC_SSD8)]
SC_arr_average = [np.mean(SC2), np.mean(SC4), np.mean(SC6), np.mean(SC8)]
#print "linked", linked
#labelList = range(1, 11)
#print "labelList", labelList
#plt.figure(figsize=(10, 7))  
#dendrogram(linked_avg)
n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)

plt.plot( WC_SSD_arr_average, '-bo', label='within-cluster sum of squared distances (WC-SSD)') 
#plt.plot( SC_arr_average, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('WC-SSD as a function of K for Average linkage')
plt.xticks(index, ('2', '4', '6', '8'))
plt.legend()
plt.tight_layout()

n_groups = 4
fig, ax = plt.subplots()
index = np.arange(n_groups)

#plt.plot( WC_SSD_arr_average, '-bo', label='within-cluster sum of squared distances (WC SSD)') 
plt.plot( SC_arr_average, '-g^', label='silhouette coefficient (SC)')
plt.xlabel('K')
#plt.ylabel('Model accuracy')
plt.title('SC as a function of K for Average linkage')
plt.xticks(index, ('2', '4', '6', '8'))
plt.legend()
plt.tight_layout()

plt.show() 

