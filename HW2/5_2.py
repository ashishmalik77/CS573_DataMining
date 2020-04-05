import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt
import random
import math



bin_size = [2, 5, 10, 50, 100, 200]

def separate_dataset_instances(dataset):
	separated_instance = {}
	for i in range(len(dataset)):
		class_value = dataset[i]
		if (class_value[-1] not in separated_instance):
			separated_instance[class_value[-1]] = []
		separated_instance[class_value[-1]].append(class_value)
	return separated_instance


def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def attribute_summary(dataset):
	separated = separate_dataset_instances(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	denom1 = (2*math.pow(stdev,2))
	#print "denom1", denom1
	if denom1 == 0.0:
		denom1 = 0.00001
	exponent = math.exp(-(math.pow(x-mean,2)/(denom1))) 
	denom2 = (math.sqrt(2*math.pi) * stdev)
	if denom2 == 0.0:
		denom2 = 0.00001
	return (1 / denom2) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def nbc(t_frac, df): #Naive Bayes 
	############ Split Dataset #########
	#test dataset
	test_data = df.sample(frac=0.2, random_state = 47)
	#print test_data.head(10)
	#print len(test_data)
	#training dataset
	training_data=df.drop(test_data.index) 
	sampling = training_data.sample(frac=t_frac, random_state = 47) #training sample
	xtrain = sampling.values[1:, 1:54]
	xtest = test_data.values[1:, 1:54]
	# prepare model
	summaries = attribute_summary(xtrain) #ok
	# train model
	predictions_train = getPredictions(summaries, xtrain)
	accuracy_train = getAccuracy(xtrain, predictions_train)
	#print('Training Accuracy: {0}%').format(round(accuracy_train,2))


	# test model
	predictions_test = getPredictions(summaries, xtest)
	accuracy_test = getAccuracy(xtest, predictions_test)
	#print('Testing Accuracy: {0}%').format(round(accuracy_test,2))
	return accuracy_train, accuracy_test
arr_train = []
arr_test = []
for bins in range(0,len(bin_size)):
	# importing data
	df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')
	#print df.head(10)
	for j in range(0, df.shape[0]):
		if df.gaming.values[j] > 10:
			df.gaming.values[j] = 10
		if df.reading.values[j] > 10:
			df.reading.values[j] = 10
	for i in df:
		#print i	
		if i != 'Unnamed: 0' and i != 'gender' and i != 'race' and i != 'race_o' and i!= 'samerace' and i!='field' and i != 'decision' and i!='attractive_important' and i!='ambition_partner' and i!='shared_interests_partner' and i!='museums' and i!='tv' and i !='interests_correlate':

			df[i] = pd.cut(df[i],bin_size[bins], labels = False)

		if i=='attractive_important' or i=='ambition_partner' or i=='shared_interests_partner' or i=='museums' or i=='tv' or i =='interests_correlate':
			df[i] = pd.cut(df[i],bin_size[bins], labels = False)
			
	#print df.attractive_important.tail(10)
	print "Bin size: ", bin_size[bins]
	accuracy_train, accuracy_test = nbc(1, df)
	print('Training Accuracy: {0}%').format(round(accuracy_train,2))
	arr_train.append(accuracy_train)
	print('Testing Accuracy: {0}%').format(round(accuracy_test,2))
	arr_test.append(accuracy_test)

#print arr_train, arr_test
# create plot
n_groups = 6
fig, ax = plt.subplots()
index = np.arange(n_groups)
'''
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, arr_train, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 label='Training Accuracy')
	 
rects2 = plt.bar(index + bar_width, arr_test, bar_width,
	                 alpha=opacity,
	                 color='g',
	                 label='Testing Accuracy')
'''
plt.plot( arr_train, '-bo', label='Training Accuracy') 
plt.plot( arr_test, '-g^', label='Testing Accuracy')
plt.xlabel('Bin size (b)')
plt.ylabel('Model accuracy')
plt.title('Effects of bin size (b) on the learned NBC model performance')
plt.xticks(index, ('2', '5', '10', '50','100','200'))
plt.legend()
plt.tight_layout()
plt.show()
