import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys


# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')
#print df.values[0]
# using encoder to read data properly without error
#print df.head(20)
#print df.dtypes


#print df.race
#print df['race']

#convert object to string and replace single quote
#df['race'] = df['race'].str.replace("'"," ")
#df['race_o'] = df['race_o'].str.replace("'"," ")
#df['field'] = df['field'].str.replace("'"," ")


#print df['race'].str.count("'") #will show in every row

#print df.shape[0] #6744

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
print "Quotes removed from ", count_race + count_race_o + count_field," cells."
print "Standardized ", count_field_lower," cells to lower case."


########################## 1) iii) Label encoding #######################


#print df.apply(lambda x: len(x.unique()))  #count the distinct elements in DataFrame in each column


for i in range (0,df.shape[0]): #removing spaces from data
	df.gender.values[i] = df.gender.values[i].strip()
unq_gender_count = len(df.gender.unique())
unq_gender_val =  df.gender.unique()
#for i in range (0,unq_gender_count): 
	#unq_gender_val[i] = unq_gender_val[i].strip()

unq_gender_val_sorted = sorted(unq_gender_val)
#flag_gender = 0
for i in range (0,df.shape[0]):
	for j in range (0,unq_gender_count):
		if df.gender.values[i] == unq_gender_val_sorted[j]:
			if df.gender.values[i] == 'male':
				flag_gender = j
			df.gender.values[i] = j
			break
#print unq_gender_count, unq_gender_val_sorted, unq_gender_val[0], unq_gender_val_sorted[0]
#print df.head(20)
print "Value assigned for male in column gender: ", flag_gender, "."


for i in range (0,df.shape[0]): #removing spaces from data
	df.race.values[i] = df.race.values[i].strip()
unq_race_count = len(df.race.unique())
unq_race_val =  df.race.unique()
#for i in range (0,unq_race_count): 
	#unq_race_val[i] = unq_race_val[i].strip()

unq_race_val_sorted = sorted(unq_race_val)
#flag_race = 0
for i in range (0,df.shape[0]):
	for j in range (0,unq_race_count):
		if df.race.values[i] == unq_race_val_sorted[j]:
			if df.race.values[i] == 'European/Caucasian-American':
				flag_race = j
			df.race.values[i] = j
			#print j
			break

#print unq_race_count, unq_race_val_sorted, unq_race_val[2], unq_race_val_sorted[2]
#print df.head(20)
print "Value assigned for European/Caucasian-American in column race: " , flag_race ,"."


for i in range (0,df.shape[0]): #removing spaces from data
	df.race_o.values[i] = df.race_o.values[i].strip()
unq_race_o_count = len(df.race_o.unique())
unq_race_o_val =  df.race_o.unique()
#for i in range (0,unq_race_o_count): #removing spaces from data
	#unq_race_o_val[i] = unq_race_o_val[i].strip()
unq_race_o_val_sorted = sorted(unq_race_o_val)
for i in range (0,df.shape[0]):
	for j in range (0,unq_race_o_count):
		if df.race_o.values[i] == unq_race_o_val_sorted[j]:
			if df.race_o.values[i] == 'Latino/Hispanic American':
				flag_race_o = j
			df.race_o.values[i] = j
			#print j
			break

#print unq_race_o_count, unq_race_o_val_sorted, unq_race_o_val[2], unq_race_o_val_sorted[2]
#print df.head(20)
print "Value assigned for Latino/Hispanic American in column race_o: " , flag_race_o ,"."

for i in range (0,df.shape[0]): #removing spaces from data
	df.field.values[i] = df.field.values[i].strip()
unq_field_count = len(df.field.unique())
unq_field_val =  df.field.unique()
#for i in range (0,unq_field_count): #removing spaces from data
	#unq_field_val[i] = unq_field_val[i].strip()
unq_field_val_sorted = sorted(unq_field_val)
field_numeric = []
for i in range (0,unq_field_count):
	field_numeric.append(i)
for i in range (0,df.shape[0]):
	for j in range (0,unq_field_count):
		if df.field.values[i] == unq_field_val_sorted[j]:
			if df.field.values[i] == 'law':
				flag_field = j
			df.field.values[i] = j
			#print j
			break
#print unq_field_count, unq_field_val_sorted, unq_field_val[2], unq_field_val_sorted[2]
#print df.head(20)
#print df.field
print "Value assigned for law in column field: " , flag_field ,"."

########################## 1) iv) Mean ##################

############### pre-process step for values in preference scores of participant ################

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

print "Mean of attractive_important: ", round(df.attractive_important.mean(),2)
print "Mean of sincere_important: ", round(df.sincere_important.mean(),2)
print "Mean of intelligence_important: ",round(df.intelligence_important.mean(),2)
print "Mean of funny_important: ", round(df.funny_important.mean(),2)
print "Mean of ambition_important: ", round(df.ambition_important.mean(),2)
print "Mean of shared_interests_important: ", round(df.shared_interests_important.mean(),2)

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

print "Mean of pref_o_attractive: ", round(df.pref_o_attractive.mean(),2)
print "Mean of pref_o_sincere: ", round(df.pref_o_sincere.mean(),2)
print "Mean of pref_o_intelligence: ",round(df.pref_o_intelligence.mean(),2)
print "Mean of pref_o_funny: ", round(df.pref_o_funny.mean(),2)
print "Mean of pref_o_ambitious: ", round(df.pref_o_ambitious.mean(),2)
print "Mean of pref_o_shared_interests:", round(df.pref_o_shared_interests.mean(),2)

#outfile = open(sys.argv[2], 'w')
#outfile.write(str(df.copy()))

df.to_csv(sys.argv[2])

