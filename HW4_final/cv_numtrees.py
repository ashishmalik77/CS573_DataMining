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
from scipy import stats
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
    
trial = np.sqrt(10)

####### Scratch ########

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



max_depth = 8
min_size = 50
#tree = build_tree(df1.values.tolist(), 8, 50)
    
    

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

#######.............. 5 The Infuence of Number of Trees on classifier Performance..........#################  

number_trees = [10, 20, 40, 50]



#.................. BT ................#
avg_accuracy_train_BT = []
std_accuracy_train_BT = []
sterr_train_BT = []
avg_accuracy_test_BT = []
std_accuracy_test_BT = []
sterr_test_BT = []
validation_BT = []
for i in number_trees:
    accuracy_train_BT_arr = []
    accuracy_test_BT_arr = []
    for j in range (1,11):
        train_set, test_set = union_trainset(j, S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10)
        predictions_train_cv_4, predictions_test_cv_4 = bagging(train_set.values.tolist(), test_set.values.tolist(), 8, 50, i)
        label_train_4 = train_set.values[0:, 49:50]
        label_test_4 = test_set.values[0:, 49:50]

        train_accuracy_BT_4 = calculate_accuracy(predictions_train_cv_4, label_train_4)
        accuracy_train_BT_arr.append(train_accuracy_BT_4)
        test_accuracy_BT_4 = calculate_accuracy(predictions_test_cv_4, label_test_4)
        accuracy_test_BT_arr.append(test_accuracy_BT_4)
        validation_BT.append(test_accuracy_BT_4)

    avg_accuracy_train_BT.append((np.mean(accuracy_train_BT_arr)))
    std_accuracy_train_BT.append(np.std(accuracy_train_BT_arr))
    avg_accuracy_test_BT.append((np.mean(accuracy_test_BT_arr)))
    std_accuracy_test_BT.append(np.std(accuracy_test_BT_arr))

for i in range (0,len(std_accuracy_train_BT)):
    sterr_train_BT.append(float(std_accuracy_train_BT[i])/trial)
    sterr_test_BT.append(float(std_accuracy_test_BT[i])/trial)
'''

print "avg_accuracy_train_BT", avg_accuracy_train_BT
print "std_accuracy_train_BT", std_accuracy_train_BT
print "sterr_train_BT", sterr_train_BT
print "avg_accuracy_test_BT (validation accuracy)", avg_accuracy_test_BT
print "std_accuracy_test_BT", std_accuracy_test_BT
print "sterr_test_BT", sterr_test_BT
'''
#.................. RF ................#


avg_accuracy_train_RF = []
std_accuracy_train_RF = []
sterr_train_RF = []
avg_accuracy_test_RF = []
std_accuracy_test_RF = []
sterr_test_RF = []
validation_RF = []

for i in number_trees:
    accuracy_train_RF_arr = []
    accuracy_test_RF_arr = []
    for j in range (1,11):
        train_set, test_set = union_trainset(j, S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10)
        predictions_train_cv_4, predictions_test_cv_4 = random_forest(train_set.values.tolist(), test_set.values.tolist(), 8, 50, i, n_features)
        label_train_4 = train_set.values[0:, 49:50]
        label_test_4 = test_set.values[0:, 49:50]

        train_accuracy_RF_4 = calculate_accuracy(predictions_train_cv_4, label_train_4)
        accuracy_train_RF_arr.append(train_accuracy_RF_4)
        test_accuracy_RF_4 = calculate_accuracy(predictions_test_cv_4, label_test_4)
        accuracy_test_RF_arr.append(test_accuracy_RF_4)
        validation_RF.append(test_accuracy_RF_4)

    avg_accuracy_train_RF.append((np.mean(accuracy_train_RF_arr)))
    std_accuracy_train_RF.append(np.std(accuracy_train_RF_arr))
    avg_accuracy_test_RF.append((np.mean(accuracy_test_RF_arr)))
    std_accuracy_test_RF.append(np.std(accuracy_test_RF_arr))

for i in range (0,len(std_accuracy_train_RF)):
    sterr_train_RF.append(float(std_accuracy_train_RF[i])/trial)
    sterr_test_RF.append(float(std_accuracy_test_RF[i])/trial)

########### Hypothesis testing BT vs RF ###########
p_value1 = stats.ttest_rel (validation_BT[0:10],validation_RF[0:10])
p_value2 = stats.ttest_rel (validation_BT[10:20],validation_RF[10:20])
p_value3 = stats.ttest_rel (validation_BT[20:30],validation_RF[20:30])
p_value4 = stats.ttest_rel (validation_BT[30:40],validation_RF[30:40])

print "p value BT and RF for number of trees 10 : ", p_value1
print "p value BT and RF for number of trees 20 : ", p_value2
print "p value BT and RF for number of trees 40 : ", p_value3
print "p value BT and RF for number of trees 50 : ", p_value4


#plt.errorbar(number_trees, avg_accuracy_test_DT, yerr=sterr_test_DT, marker='o', color='blue',ecolor='gray',elinewidth=1, capsize=2, barsabove = True, label='Validation Accuracy DT')
plt.errorbar(number_trees, avg_accuracy_test_BT, yerr=sterr_test_BT, marker='o', color='green',ecolor='purple',elinewidth=1, capsize=2, barsabove = True, label='Validation Accuracy BT')
plt.errorbar(number_trees, avg_accuracy_test_RF, yerr=sterr_test_RF, marker='o', color='red',ecolor='black',elinewidth=1, capsize=2, barsabove = True, label='Validation Accuracy RF')
plt.xlabel('Number of trees')
plt.ylabel('Model accuracy')
plt.title('Learning curve')
#plt.xticks(index, ('0.01', '0.1', '0.2', '0.5', '0.6', '0.75', '0.9', '1'))
plt.xlim((1,60))                 #Set X-axis limits
#plt.xticks(np.arange(len(size_train_data)), size_train_data[0:6])
#plt.errorbar(size_train_data, avg_accuracy_test_nbc, yerr=sterr_test_nbc, fmt='o', color='black',ecolor='lightgray', elinewidth=3, capsize=0);
plt.legend()
#plt.tight_layout()
plt.show()

