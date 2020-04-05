import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys


# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
df1 = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')
#print df.values[0]
# using encoder to read data properly without error
#print df1.shape # (row, column) (6744, 53)
#print df.dtypes
#after dropping last 244 rows
df = df1.head(6500)
#print df.shape #6500 


########################## 1) i) remove quotes #######################

count_race = 0
for i in range(0,df.shape[0]):
	if "'" in df['race'].values[i]:
		count_race = count_race + 1
		df['race'].values[i] = df['race'].values[i].replace("'"," ")
#print count_race


count_race_o = 0
for i in range(0,df.shape[0]):
	if "'" in df['race_o'].values[i]:
		count_race_o = count_race_o + 1
		df['race_o'].values[i] = df['race_o'].values[i].replace("'"," ")
#print count_race_o



count_field = 0
for i in range(0,df.shape[0]):
	if "'" in df['field'].values[i]:
		count_field = count_field + 1
		df['field'].values[i] = df['field'].values[i].replace("'"," ")
#print count_field

#print df.field

########################## 1) ii) convert lowercase #######################

#df['field'].values[0] = df['field'].values[0].lower()
count_field_lower = 0
for i in range(0,df.shape[0]):
	if any(df['field'].values[i][j].isupper() for j in range(0,len(df['field'].values[i]))):
		count_field_lower = count_field_lower + 1
		df['field'].values[i] = df['field'].values[i].lower()
#print count_field_lower

#print df.field
#print df.head(20)
#print "Quotes removed from ", count_race + count_race_o + count_field," cells."
#print "Standardized ", count_field_lower," cells to lower case."


########################## 1) iii) One-hot encoding #######################



for i in range (0,df.shape[0]): #removing spaces from data
	df.gender.values[i] = df.gender.values[i].strip()
unq_gender_count = len(df.gender.unique())
unq_gender_val =  df.gender.unique()
unq_gender_val_sorted = sorted(unq_gender_val)
#print unq_gender_count, unq_gender_val, unq_gender_val_sorted #ok

for i in range (0,df.shape[0]): #removing spaces from data
	df.race.values[i] = df.race.values[i].strip()
unq_race_count = len(df.race.unique())
unq_race_val =  df.race.unique()
unq_race_val_sorted = sorted(unq_race_val)
#print unq_race_count, unq_race_val, unq_race_val_sorted  #ok

for i in range (0,df.shape[0]): #removing spaces from data
	df.race_o.values[i] = df.race_o.values[i].strip()
unq_race_o_count = len(df.race_o.unique())
unq_race_o_val =  df.race_o.unique()
unq_race_o_val_sorted = sorted(unq_race_o_val)
#print unq_race_o_count, unq_race_o_val, unq_race_o_val_sorted #ok


for i in range (0,df.shape[0]): #removing spaces from data
	df.field.values[i] = df.field.values[i].strip()
unq_field_count = len(df.field.unique())
unq_field_val =  df.field.unique()
unq_field_val_sorted = sorted(unq_field_val)
#print unq_field_count, unq_field_val, unq_field_val_sorted #ok

'''
print df.head(10)
df = pd.get_dummies(data=df, columns=['gender', 'race','race_o','field'])
print df.head(10)
print df.shape
'''

'''
S = pd.Series( {'A': ['b', 'a', 'c']})
print(S)
one_hot = pd.get_dummies(S['A'])
print(one_hot)
'''

#print those mapped vector

S = pd.Series( {'gender': unq_gender_val })
#print(S)
one_hot = pd.get_dummies(S['gender'])
'''
print(one_hot)
one_hot.drop(columns = ['male'], inplace=True) #drop last column
print(one_hot)
one_hot.drop(one_hot.tail(1).index,inplace=True) # drop last row
print(one_hot)
'''
female_arr = []
for i in range (0, one_hot.shape[0] - 1):
	female_arr.append(one_hot.female.values[i])
print "Mapped vector for female in column gender: ", female_arr



S = pd.Series( {'race': unq_race_val_sorted})
#print(S)
one_hot = pd.get_dummies(S['race'])
#print(one_hot['Black/African American'])
blck_aa_arr = []
for i in range (0, one_hot.shape[0] - 1):
	blck_aa_arr.append(one_hot['Black/African American'].values[i])
print "Mapped vector for Black/African American in column race: ", blck_aa_arr


S = pd.Series( {'race_o': unq_race_o_val_sorted})
#print(S)
one_hot = pd.get_dummies(S['race_o'])
#print(one_hot.Other)
other_arr = []
for i in range (0, one_hot.shape[0]-1):
	other_arr.append(one_hot['Other'].values[i])
print "Mapped vector for Other in column race_o: ", other_arr


S = pd.Series( {'field': unq_field_val_sorted})
#print(S)
one_hot = pd.get_dummies(S['field'])
#print(one_hot.economics)
economics_arr = []
for i in range (0, one_hot.shape[0]-1):
	economics_arr.append(one_hot['economics'].values[i])
print "Mapped vector for economics in column field: ", economics_arr

#print df.head(10)
df = pd.get_dummies(data=df, columns=['gender', 'race','race_o','field'])
#print df.shape # before dropping number of rows = 6500 , columns = 265
df.drop(columns = ['gender_male'], axis=1, inplace=True)
df.drop(columns = ['race_Other'], axis=1, inplace=True)
df.drop(columns = ['race_o_Other'], axis=1, inplace=True)
df.drop(columns = ['field_writing: literary nonfiction'], axis=1, inplace=True)
#print df.head(10)
#print df.shape  # number of rows = 6500 , columns = 261
#print df.columns

########################## 1) iv) Mean ##################

############### pre-process step for values in preference scores of participant ################
pd.set_option('mode.chained_assignment', None) #avoid the warning
for i in range(0,df.shape[0]):
	#print df.attractive_important.values[i]
	total = df.attractive_important.values[i] + df.sincere_important.values[i] + df.intelligence_important.values[i]+df.funny_important.values[i]+df.ambition_important.values[i]+df.shared_interests_important.values[i]
	#print "total: ", total
	df.attractive_important.values[i] = df.attractive_important.values[i]/total
	df.sincere_important.values[i] = df.sincere_important.values[i]/total
	df.intelligence_important.values[i] = df.intelligence_important.values[i]/total
	df.funny_important.values[i] = df.funny_important.values[i]/total
	df.ambition_important.values[i] = df.ambition_important.values[i]/total
	df.shared_interests_important[i] = df.shared_interests_important[i]/total
'''
print "Mean of attractive_important: ", round(df.attractive_important.mean(),2)
print "Mean of sincere_important: ", round(df.sincere_important.mean(),2)
print "Mean of intelligence_important: ",round(df.intelligence_important.mean(),2)
print "Mean of funny_important: ", round(df.funny_important.mean(),2)
print "Mean of ambition_important: ", round(df.ambition_important.mean(),2)
print "Mean of shared_interests_important: ", round(df.shared_interests_important.mean(),2)
'''
############### pre-process step for values in preference scores of partner ################
for i in range(0,df.shape[0]):
	#print df.pref_o_attractive.values[i]
	total_pref = df.pref_o_attractive.values[i] + df.pref_o_sincere.values[i] + df.pref_o_intelligence.values[i]+df.pref_o_funny.values[i]+df.pref_o_ambitious.values[i]+df.pref_o_shared_interests.values[i]
	#print "total: ", total
	df.pref_o_attractive.values[i] = df.pref_o_attractive.values[i]/total_pref
	df.pref_o_sincere.values[i] = df.pref_o_sincere.values[i]/total_pref
	df.pref_o_intelligence.values[i] = df.pref_o_intelligence.values[i]/total_pref
	df.pref_o_funny.values[i] = df.pref_o_funny.values[i]/total_pref
	df.pref_o_ambitious.values[i] = df.pref_o_ambitious.values[i]/total_pref
	df.pref_o_shared_interests[i] = df.pref_o_shared_interests[i]/total_pref
'''
print "Mean of pref_o_attractive: ", round(df.pref_o_attractive.mean(),2)
print "Mean of pref_o_sincere: ", round(df.pref_o_sincere.mean(),2)
print "Mean of pref_o_intelligence: ",round(df.pref_o_intelligence.mean(),2)
print "Mean of pref_o_funny: ", round(df.pref_o_funny.mean(),2)
print "Mean of pref_o_ambitious: ", round(df.pref_o_ambitious.mean(),2)
print "Mean of pref_o_shared_interests:", round(df.pref_o_shared_interests.mean(),2)
'''

df['decision_dup'] = df['decision']
#print df.shape # 6500, 262 (created duplicate column of decision column as decision_dup)
df.drop(columns = ['decision'], axis=1, inplace=True)
#print df.shape # 6500, 261 #dropped decision 

test_data = df.sample(frac=0.2, random_state = 25)
#training_data = df.sample(frac=0.8, random_state = 25) # not this way
#print test_data.index
training_data=df.drop(test_data.index) 
#print training_data.shape #5200 rows x 261 columns
#print test_data.shape #1300 rows x 261 columns

training_data.to_csv(sys.argv[2])
test_data.to_csv(sys.argv[3])


#outfile = open(sys.argv[2], 'w')
#outfile.write(str(df.copy()))

######################################### Split Dataset ######################
#df.to_csv(sys.argv[2])

