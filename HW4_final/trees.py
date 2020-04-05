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
modelIdx = sys.argv[3]
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
    

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    #print "hi test_split"
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    #print "left :", left
    #print "right :", right
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    #print "hi gini_index"
    n_samples = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_samples)
    #print "bye gini_index"
    #print gini
    return gini


# Select the best split point for a dataset

def get_split(dataset):
    #print "hi get_split"
    #c = 0
    class_values = list(set(row[-1] for row in dataset))
    #print "class_values",class_values #[0,1]
    best_feature = 10000
    best_value = 10000
    best_score = 10000
    best_groups = None
    #print "len(dataset[0])-1", len(dataset[0])-1 #49
    for feature in range(len(dataset[0])-1):
        for row in dataset:
            #print "row", row
            #print "feature", feature
            groups = test_split(feature, row[feature], dataset)
            #print "groups", groups
            gini = gini_index(groups, class_values)
            #c = c+ 1
            
            if gini < best_score:
                best_feature = feature
                best_value = row[feature]
                best_score = gini
                best_groups = groups
            else:
                break
    #print "count ",c
    output = {}
    output['feature']= best_feature
    output['value']= best_value
    output['groups'] = best_groups
    return output

# leaf node
def leaf(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    #print "hi needed split"
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = leaf(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = leaf(left), leaf(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = leaf(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = leaf(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
 

# Build a decision tree
def build_tree(train, max_depth, min_size):
    #print "hi build tree"
    #print "len(train)",len(train) #5200
    root = get_split(train)
    #print "got root", root, type(root)
    split(root, max_depth, min_size, depth =1)
    return root




#tree = build_tree(df1.values[0:, 0:50], 8, 50)

#print type(dataset)
#print type(df1.values[0:, 0:50])
#tree = build_tree(df1.values.tolist(), 8, 50)
#print_tree(tree)
# Make a prediction with a decision tree
def predict(node, row):
    if row[node['feature']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    #print ("type tree"), type(tree), tree
    predictions_train = list()
    for row_train in train:
        prediction_train = predict(tree, row_train)
        predictions_train.append(prediction_train)
    #print "predictions_train", predictions_train, len(predictions_train)
    
    predictions_test = list()
    for row in test:
        prediction_test = predict(tree, row)
        predictions_test.append(prediction_test)
    #print "Y_test", Y_test, type(Y_test)
    
    return predictions_train, predictions_test


def calculate_accuracy(pred, label):
    count = 0
    for i in range (0, len(pred)):
        if pred[i] == label[i]:
            count+= 1
    accuracy = float(count) / len(pred)
    return accuracy


max_depth = 8
min_size = 50
#tree = build_tree(df1.values.tolist(), 8, 50)

if modelIdx == '1': #DT
    predictions_train, predictions_test = decision_tree(df1.values.tolist(), df2.values.tolist(), max_depth, min_size)
    train_accuracy_DT = calculate_accuracy(predictions_train, Y_train)
    '''
    count_train = 0
    for i in range (0, len(predictions_train)):
        if predictions_train[i] == Y_train[i]:
            count_train+= 1
    train_accuracy_DT = float(count_train) / len(predictions_train)
    '''
    print 'Training Accuracy DT:', round(train_accuracy_DT,2)
    test_accuracy_DT = calculate_accuracy(predictions_test, Y_test)
    '''
    count_test = 0
    for i in range (0, len(predictions_test)):
        if predictions_test[i] == Y_test[i]:
            count_test+= 1
    test_accuracy_DT = float(count_test) / len(predictions_test)
    #test_accuracy_DT = (predictions_test == Y_test).sum().astype(float) / len(predictions_test)
    '''
    print 'Test Accuracy DT:', round(test_accuracy_DT,2)
    
    

##########..................Bagging from Scratch.....................#########

# Create pseudosamples from the dataset with replacement
def pseudosamples(dataset):
    sample = list()
    n_sample = len(dataset)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample
 
# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)
 
# Bootstrap Aggregation Algorithm
def bagging(train, test, max_depth, min_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = pseudosamples(train)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions_train = [bagging_predict(trees, row_train) for row_train in train]
    predictions_test = [bagging_predict(trees, row_test) for row_test in test]
    return predictions_train, predictions_test


n_trees = 30

if modelIdx == '2': #BT
    predictions_train_BT, predictions_test_BT = bagging(df1.values.tolist(), df2.values.tolist(), max_depth, min_size, n_trees)
    train_accuracy_BT = calculate_accuracy(predictions_train_BT, Y_train)
    '''
    count_train_BT = 0
    for i in range (0, len(predictions_train_BT)):
        if predictions_train_BT[i] == Y_train[i]:
            count_train_BT+= 1
    train_accuracy_BT = float(count_train_BT) / len(predictions_train_BT)
    '''
    print 'Train Accuracy BT:', round(train_accuracy_BT,2)
    test_accuracy_BT = calculate_accuracy(predictions_test_BT, Y_test)

    '''
    count_test_BT = 0
    for i in range (0, len(predictions_test_BT)):
        if predictions_test_BT[i] == Y_test[i]:
            count_test_BT+= 1
    test_accuracy_BT = float(count_test_BT) / len(predictions_test_BT)
    '''
    print 'Test Accuracy BT:', round(test_accuracy_BT,2)

###########.......Random Forest............#############

def split_RF(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = leaf(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = leaf(left), leaf(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = leaf(left)
    else:
        node['left'] = get_split_RF(left, n_features)
        split_RF(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = leaf(right)
    else:
        node['right'] = get_split_RF(right, n_features)
        split_RF(node['right'], max_depth, min_size, n_features, depth+1)
 

def get_split_RF(dataset, n_features):
    #print "hi get_split_RF"
    class_values = list(set(row[-1] for row in dataset))
    best_feature = 10000
    best_value = 10000
    best_score = 10000
    best_groups = None
    features = list()
    while len(features) < n_features:
        feature_rf = randrange(len(dataset[0])-1)
        #print "feature_rf", feature_rf
        if feature_rf not in features:
            features.append(feature_rf)
    #print "features", features
    for feature_rf in features:
        for row in dataset:
            groups = test_split(feature_rf, row[feature_rf], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_feature = feature_rf
                best_value = row[feature_rf]
                best_score = gini
                best_groups = groups
            else:
                break
    output = {}
    output['feature']= best_feature
    output['value']= best_value
    output['groups'] = best_groups
    return output
    


def build_tree_RF(train, max_depth, min_size, n_features):
    #print "hi build_tree_RF"
    root = get_split_RF(train, n_features)
    #print "got root"
    split_RF(root, max_depth, min_size, n_features, 1)
    #print "split_RF"
    return root
 

def random_forest(train, test, max_depth, min_size, n_trees, n_features):
    #print "hi RF"
    trees = list()
    for i in range(n_trees):
        sample = pseudosamples(train)
        tree = build_tree_RF(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions_train = [bagging_predict(trees, row) for row in train]
    predictions_test = [bagging_predict(trees, row) for row in test]
    #print "bye RF"
    return predictions_train, predictions_test




n_features = int(np.sqrt(df1.shape[1]-1))
#print "n_features", n_features

if modelIdx == '3': #RF
    predictions_train_RF, predictions_test_RF = random_forest(df1.values.tolist(), df2.values.tolist(), max_depth, min_size, n_trees, n_features)
    train_accuracy_RF = calculate_accuracy(predictions_train_RF, Y_train)
    print 'Train Accuracy RF:', round(train_accuracy_RF,2)
    test_accuracy_RF = calculate_accuracy(predictions_test_RF, Y_test)
    print 'Test Accuracy RF:', round(test_accuracy_RF,2)  


