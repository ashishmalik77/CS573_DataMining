import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt
import random
import math
from scipy import optimize as op  
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None) 
from random import randrange


# importing data
df1 = pd.read_csv( sys.argv[1], encoding='ISO-8859-1') #training set
df2 = pd.read_csv( sys.argv[2], encoding='ISO-8859-1') #test set
#print df1.shape # (5200, 51)
#print df2.shape # (1300, 51)
#print df1.head(10)
df1.drop(columns = ['Unnamed: 0'], axis=1,  inplace=True)
df2.drop(columns = ['Unnamed: 0'], axis=1,  inplace=True)
#print df1.shape # (5200, 50)
#print df2.shape # (1300, 50)
#print df1.head(10)
#print df2.head(10)
#modelIdx = sys.argv[3]
#print "modelIdx",modelIdx
##########..................Decision Trees.....................
X_train = df1.values[0:, 0:49]
Y_train = df1.values[0:, 49:50]
#print "X_train", type(X_train), X_train
#print Y_train, len(Y_train)
#y = (df1.iloc[:, 261:262]).values
#print y, len(y)
#y_train = y_train.ravel()
X_test = df2.values[0:, 0:49]
Y_test = df2.values[0:, 49:50]


shuffle_train = df1.sample(frac=1.0, random_state = 18)
#print shuffle_train.shape, shuffle_train.head(10) #(5200, 50)  
sample_train = shuffle_train.sample(frac=0.5, random_state = 32)
#print sample_train.shape, sample_train.head(10) #(2600, 50)  
#1st_fold = sample_train.sample(frac=0.1)
S1 = sample_train.values[0:260, 0:50]
S2 = sample_train.values[260:520, 0:50]
S3 = sample_train.values[520:780, 0:50]
S4 = sample_train.values[780:1040, 0:50]
S5 = sample_train.values[1040:1300, 0:50]
S6 = sample_train.values[1300:1560, 0:50]
S7 = sample_train.values[1560:1820, 0:50]
S8 = sample_train.values[1820:2080, 0:50]
S9 = sample_train.values[2080:2340, 0:50]
S10 = sample_train.values[2340:2600, 0:50]  
    
def union_trainset(j, S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10):
    train_set_union = []
    #print "j",j
    if j == 1:
        test_set = pd.DataFrame(S1)
        #print "S1", S1, type(S1), type(test_set) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 2:
        test_set = pd.DataFrame(S2)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 3:
        test_set = pd.DataFrame(S3)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S4 , S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 4:
        test_set = pd.DataFrame(S4)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1, S2 , S3, S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 5:
        test_set = pd.DataFrame(S5)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2, S3 , S4, S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 6:
        test_set = pd.DataFrame(S6)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S3, S4 , S5, S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 7:
        test_set = pd.DataFrame(S7)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1, S2 , S3 , S4 , S5 , S6, S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 8:
        test_set = pd.DataFrame(S8)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2, S3 , S4 , S5 , S6 , S7, S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 9:
        test_set = pd.DataFrame(S9)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S3, S4 , S5 , S6 , S7 , S8 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 10:
        test_set = pd.DataFrame(S10)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S3, S4 , S5 , S6 , S7 , S8 , S9), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        #train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    return train_set_union_df, test_set


feature_set = X_train
labels = Y_train.reshape(Y_train.shape[0],1)  
N = Y_train.shape[0]
test_label = Y_test.reshape(Y_test.shape[0],1) 
#labels = labels.reshape(5200,1)
#reg_lambda = 0.01 # 63%
#reg_lambda = 0.001 #64#
reg_lambda = 0.0001 #74%

np.random.seed(1)  
weights = np.random.rand(feature_set.shape[1],1)  

bias = np.random.rand(1)  
#lr = 0.09 #63%
#lr = 0.01 #64#
lr = 0.0001 #74% 1000 iteration 75%

def sigmoid(z):  
    return 1/(1+np.exp(-z))

def sigmoid_der(z):  
    return sigmoid(z)*(1-sigmoid(z))
    #return np.exp(-z) / ( (1+np.exp(-z)) * (1+np.exp(-z)))

def neural_network(weights, bias, feature_set, labels):
    for epoch in range(1000):  

        # feedforward step1
        XW = np.dot(feature_set, weights) + bias

        #feedforward step2
        z = sigmoid(XW)
        #print "z", z


        # backpropagation step 
        dcost_dpred = z - labels

        dpred_dz = sigmoid_der(z)

        z_delta = dcost_dpred * dpred_dz


        # Add regularization terms (b1 and b2 don't have regularization terms) 
        
        gradient = np.dot(feature_set.T, z_delta) + ((reg_lambda * weights) ) 
 
        weights -= lr * gradient

        for num in z_delta:
            bias -= lr * num
    #print "weights", weights
    #print "bias", bias
    return weights, bias

weights, bias = neural_network(weights, bias, feature_set, labels)

final_scores = np.dot(feature_set, weights) + bias
preds = np.round(sigmoid(final_scores))
train_accuracy_LR = (preds == labels).sum().astype(float) / len(preds)
print 'Training Accuracy NN:', round(train_accuracy_LR,2)


# prediction for testing 
final_scores2 = np.dot(X_test, weights) + bias
preds2 = np.round(sigmoid(final_scores2))
test_accuracy_LR = (preds2 == test_label).sum().astype(float) / len(preds2) 
print 'Testing Accuracy NN:',round(test_accuracy_LR,2)

####### 10 Fold cross Validation ##########

def calculate_accuracy(pred, label):
    count = 0
    for i in range (0, len(pred)):
        if pred[i] == label[i]:
            count+= 1
    accuracy = float(count) / len(pred)
    return accuracy


test_accuracy_NN_arr = []
for j in range (1,11):
    train_set, test_set = union_trainset(j, S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10)
    #label_train_3 = train_set.values[0:, 49:50]
    feature_test = test_set.values[0:, 0:49]
    label_test = test_set.values[0:, 49:50]
    label_test_rs = label_test.reshape(label_test.shape[0],1) 
    weights_cv, bias_cv = neural_network(weights, bias, feature_test, label_test_rs)
    final_scores2 = np.dot(feature_test, weights_cv) + bias_cv
    preds2 = np.round(sigmoid(final_scores2))
    test_accuracy_NN = calculate_accuracy(preds2, label_test_rs)
    
    print '10 Fold Validation Accuracy NN:',round(test_accuracy_NN,2)
    test_accuracy_NN_arr.append(test_accuracy_NN)

print "Average 10-fold cross validation accuracy of NN: ", round(np.mean(test_accuracy_NN_arr),2)
