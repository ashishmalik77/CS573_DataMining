import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt

# importing data
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')

#print df.values[0]


test_data = df.sample(frac=0.2, random_state = 47)
#training_data = df.sample(frac=0.8, random_state = 47) # not this way
training_data=df.drop(test_data.index) 
#print training_data.head(10) #10 rows x 55 columns
#print test_data.head(10) #10 rows x 55 columns
training_data.to_csv(sys.argv[2])
test_data.to_csv(sys.argv[3])

'''
print "len: ", len(test_data), len(training_data)
print test_data.head(10)
print training_data.head(10)
'''
