import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt
import random
import math

# importing data
df1 = pd.read_csv( sys.argv[1], encoding='ISO-8859-1') #training set
df2 = pd.read_csv( sys.argv[2], encoding='ISO-8859-1') #test set

#print df1.head(10) #10 rows x 56 columns
#print df2.head(10), df2.columns,  #10 rows x 56 columns

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


def nbc(t_frac):
	sampling = df1.sample(frac=t_frac, random_state = 47)
	xtrain = sampling.values[1:, 3:56]
	xtest = df2.values[1:, 3:56]

	# prepare model
	summaries = attribute_summary(xtrain) #ok
	# train model
	predictions_train = getPredictions(summaries, xtrain)
	accuracy_train = getAccuracy(xtrain, predictions_train)
	print('Training Accuracy: {0}%').format(round(accuracy_train,2))


	# test model
	predictions_test = getPredictions(summaries, xtest)
	accuracy_test = getAccuracy(xtest, predictions_test)
	print('Testing Accuracy: {0}%').format(round(accuracy_test,2))

nbc(1)

