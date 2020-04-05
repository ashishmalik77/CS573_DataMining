import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys


# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
pd.set_option('mode.chained_assignment', None) #avoid the warning
df1 = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')
#print df.values[0]
# using encoder to read data properly without error
#print df1.shape # (row, column) (6744, 53)
#print df.dtypes
#after dropping last 244 rows
df = df1.head(6500)
#print df.shape #(6500,53) 
##########  drop the columns race, race o and field ###############
df.drop(columns = ['race'], axis=1,  inplace=True)
df.drop(columns = ['race_o'], axis=1,  inplace=True)
df.drop(columns = ['field'], axis=1,  inplace=True)
#print df.shape #(6500, 50) 
#print df.head(10)
###########Label encoding on gender###########
label_encoding = {}

def perform_label_encoding(column):
    column = column.astype('category')
    codes_for_column = {}
    for i, category in enumerate(column.cat.categories):
        codes_for_column[category] = i
    label_encoding[column.name] = codes_for_column
    return column.cat.codes


df[['gender']] = df[['gender']].apply(perform_label_encoding)

#print('Value assigned for male in column gender:', label_encoding['gender']['male'])
#print "after label encoding:", df.head(10)
########################## Normalize preference scores of the participant ##################

columns1  = ['attractive_important', 'sincere_important', 'intelligence_important','funny_important', 'ambition_important', 'shared_interests_important']
columns2  = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence','pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
df[columns1] = df[columns1].div(df[columns1].sum(axis=1), axis=0)
df[columns2] = df[columns2].div(df[columns2].sum(axis=1), axis=0)

'''
for column in columns1:
    print('Mean of '+column+': ' + str(round(df[column].mean(),2)))

for column in columns2:
    print('Mean of '+column+': ' + str(round(df[column].mean(),2)))
'''
#print "after normaliztion:", df.head(10)
##################### Discretize all the continuous-valued columns ################

#print "before:", max(df.gaming.values), max(df.reading.values), df.reading.values[1]
for i in range(0, df.shape[0]):
	if df.gaming.values[i] > 10:
		df.gaming.values[i] = 10
	if df.reading.values[i] > 10:
		df.reading.values[i] = 10
#print "after: ", max(df.gaming.values), max(df.reading.values), df.reading.values[1]

for i in df:
	#print i	
	if  i != 'gender' and i!= 'samerace' and i != 'decision' and i!='attractive_important' and i!='ambition_partner' and i!='shared_interests_partner' and i!='museums' and i!='tv' and i !='interests_correlate':
		#print 'hi'
		df[i] = pd.cut(df[i],2,labels=[0,1])
		hist = df.groupby(i)[i].count().reset_index(name='Count').to_dict(orient='records')
		val = []
		for j in range(0,len(hist)):
			val.append(hist[j].values()[0])
		#print i,": ", val

	if i=='attractive_important' or i=='ambition_partner' or i=='shared_interests_partner' or i=='museums' or i=='tv' or i =='interests_correlate':
		df[i] = pd.cut(df[i],2,labels=[0,1])
		hist = df.groupby(i)[i].count().reset_index(name='Count').to_dict(orient='records')
		val = []
		for j in range(0,len(hist)):
			val.append(hist[j].values()[1])
		#print i,": ", val

#print "after Discretize:", df.head(10)
##########Split Dataset###########
test_data = df.sample(frac=0.2, random_state = 47)
#training_data = df.sample(frac=0.8, random_state = 25) # not this way
#print test_data.index
training_data=df.drop(test_data.index) 
#print training_data.shape #5200 rows x 50 columns
#print test_data.shape #1300 rows x 50 columns


training_data.to_csv(sys.argv[2])
test_data.to_csv(sys.argv[3])


