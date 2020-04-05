import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt

# importing data
#df = pd.read_csv('dating-full.csv', encoding='ISO-8859-1')
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')
# using encoder to read data properly without error
#print df.head(20)
#print df.dtypes



print "Number of distinct values for attractive_partner attribute: ", len(df.attractive_partner.unique())
print "Number of distinct values for sincere_partner attribute: ", len(df.sincere_partner.unique())
print "Number of distinct values for intelligence_parter attribute: ", len(df.intelligence_parter.unique())
print "Number of distinct values for funny_partner attribute: ", len(df.funny_partner.unique())
print "Number of distinct values for ambition_partner attribute: ", len(df.ambition_partner.unique())
print "Number of distinct values for shared_interests_partner attribute: ", len(df.shared_interests_partner.unique())

#print df.attractive_partner.unique()
#print df.intelligence_parter.unique()
#print df.attractive_partner.head(30)
#print df.decision.head(10)

###################### 2. (ii). (b) ##################################
count_success_attractive = 0
count_attractive_partner_10 = 0
for i in range (0, df.shape[0]):
	if df.attractive_partner.values[i] == 10 :
		count_attractive_partner_10+= 1
		if df.decision.values[i] == 1:
			count_success_attractive+=1

success_rate_attractive = float(count_success_attractive)/count_attractive_partner_10

#print count_success_attractive, count_attractive_partner_10, success_rate_attractive
print "success_rate_attractive partner: ", success_rate_attractive

###################### 2. (ii). (c) ##################################
count_success_sincere = 0
count_success_sincere_10 = 0
for i in range (0, df.shape[0]):
	if df.sincere_partner.values[i] == 10 :
		count_success_sincere_10+= 1
		if df.decision.values[i] == 1:
			count_success_sincere+=1

success_rate_sincere = float(count_success_sincere)/count_success_sincere_10

#print count_success_sincere, count_success_sincere_10, success_rate_sincere
print "success_rate_sincere :", success_rate_sincere

count_success_intelligence = 0
count_success_intelligence_10 = 0
for i in range (0, df.shape[0]):
	if df.intelligence_parter.values[i] == 10 :
		count_success_intelligence_10+= 1
		if df.decision.values[i] == 1:
			count_success_intelligence+=1

success_rate_intelligence = float(count_success_intelligence)/count_success_intelligence_10

#print count_success_intelligence, count_success_intelligence_10, success_rate_intelligence
print "success_rate_intelligence: ", success_rate_intelligence


count_success_funny = 0
count_success_funny_10 = 0
for i in range (0, df.shape[0]):
	if df.funny_partner.values[i] == 10 :
		count_success_funny_10+= 1
		if df.decision.values[i] == 1:
			count_success_funny+=1

success_rate_funny = float(count_success_funny)/count_success_funny_10
print "success_rate_funny: ", success_rate_funny



count_success_ambition = 0
count_success_ambition_10 = 0
for i in range (0, df.shape[0]):
	if df.ambition_partner.values[i] == 10 :
		count_success_ambition_10+= 1
		if df.decision.values[i] == 1:
			count_success_ambition+=1

success_rate_ambition = float(count_success_ambition)/count_success_ambition_10
print "success_rate_ambition: ", success_rate_ambition

count_success_shared_interests = 0
count_success_shared_interests_10 = 0
for i in range (0, df.shape[0]):
	if df.shared_interests_partner.values[i] == 10 :
		count_success_shared_interests_10+= 1
		if df.decision.values[i] == 1:
			count_success_shared_interests+=1

success_rate_shared_interests = float(count_success_shared_interests)/count_success_shared_interests_10
print "success_rate_shared_interests: ", success_rate_shared_interests

###################### 2. (ii). (d) ##################################

val =  df.attractive_partner.unique()
success_rate_attractive_arr = []
count_attr_total_arr = []
count_success_arr = []
#print val[0], type(val[0])
#print val
for i in range(0,len(val)):
	count_attr_total = 0
	count_success = 0
	for j in range (0, df.shape[0]): 
		if df.attractive_partner.values[j] == val[i] :
			count_attr_total+= 1
			if df.decision.values[j] == 1:
				count_success+= 1
	count_attr_total_arr.append(count_attr_total)
	count_success_arr.append(count_success)

#print "loop ", count_attr_total_arr, count_success_arr

success_rate_attractive_arr = []

for i in range(0,len(val)):
	success_rate_attractive_arr.append(float(count_success_arr[i])/count_attr_total_arr[i])
#print success_rate_attractive_arr

# Draw Scatter Plot
x = val
y = success_rate_attractive_arr
colors = (0.1,0.8,.06)
area = np.pi*3
 
# Plot
plt.scatter(x, y, s=area, c='r', alpha=0.5)
plt.title('Scatter plot of success rate for attribute attractive_partner')
plt.xlabel('attractive_partner')
plt.ylabel('Success rate')
plt.show()


val2 =  df.sincere_partner.unique()
success_rate_attractive_arr2= []
count_attr_total_arr2 = []
count_success_arr2 = []
#print val[0], type(val[0])
#print val
for i in range(0,len(val2)):
	count_attr_total2 = 0
	count_success2 = 0
	for j in range (0, df.shape[0]): 
		if df.sincere_partner.values[j] == val2[i] :
			count_attr_total2+= 1
			if df.decision.values[j] == 1:
				count_success2+= 1
	count_attr_total_arr2.append(count_attr_total2)
	count_success_arr2.append(count_success2)

#print "loop ", count_attr_total_arr, count_success_arr

success_rate_attractive_arr2 = []

for i in range(0,len(val2)):
	success_rate_attractive_arr2.append(float(count_success_arr2[i])/count_attr_total_arr2[i])
#print success_rate_attractive_arr

# Draw Scatter Plot
x2 = val2
y2 = success_rate_attractive_arr2
area = np.pi*3
 
# Plot
plt.scatter(x2, y2, c='r', s=area)
plt.title('Scatter plot of success rate for attribute sincere_partner')
plt.xlabel('sincere_partner')
plt.ylabel('Success rate')
plt.show()

val3 =  df.intelligence_parter.unique()
success_rate_attractive_arr3= []
count_attr_total_arr3 = []
count_success_arr3 = []
#print val[0], type(val[0])
#print val
for i in range(0,len(val3)):
	count_attr_total3 = 0
	count_success3 = 0
	for j in range (0, df.shape[0]): 
		if df.intelligence_parter.values[j] == val3[i] :
			count_attr_total3+= 1
			if df.decision.values[j] == 1:
				count_success3+= 1
	count_attr_total_arr3.append(count_attr_total3)
	count_success_arr3.append(count_success3)

#print "loop ", count_attr_total_arr, count_success_arr

success_rate_attractive_arr3 = []

for i in range(0,len(val3)):
	success_rate_attractive_arr3.append(float(count_success_arr3[i])/count_attr_total_arr3[i])
#print success_rate_attractive_arr

# Draw Scatter Plot
x3 = val3
y3 = success_rate_attractive_arr3
colors = (0.1,0.8,.06)
area = np.pi*3
 
# Plot
plt.scatter(x3, y3, s=area, c='r', alpha=0.5)
plt.title('Scatter plot of success rate for attribute intelligence_partner')
plt.xlabel('intelligence_partner')
plt.ylabel('Success rate')
plt.show()

val4 =  df.funny_partner.unique()
success_rate_attractive_arr4= []
count_attr_total_arr4 = []
count_success_arr4 = []
#print val[0], type(val[0])
#print val
for i in range(0,len(val4)):
	count_attr_total4 = 0
	count_success4 = 0
	for j in range (0, df.shape[0]): 
		if df.funny_partner.values[j] == val4[i] :
			count_attr_total4+= 1
			if df.decision.values[j] == 1:
				count_success4+= 1
	count_attr_total_arr4.append(count_attr_total4)
	count_success_arr4.append(count_success4)

#print "loop ", count_attr_total_arr, count_success_arr

success_rate_attractive_arr4 = []

for i in range(0,len(val4)):
	success_rate_attractive_arr4.append(float(count_success_arr4[i])/count_attr_total_arr4[i])
#print success_rate_attractive_arr

# Draw Scatter Plot
x4 = val4
y4 = success_rate_attractive_arr4
colors = (0.1,0.8,.06)
area = np.pi*3
 
# Plot
plt.scatter(x4, y4, s=area, c='r', alpha=0.5)
plt.title('Scatter plot of success rate for attribute funny_partner')
plt.xlabel('funny_partner')
plt.ylabel('Success rate')
plt.show()

val5 =  df.ambition_partner.unique()
success_rate_attractive_arr5= []
count_attr_total_arr5 = []
count_success_arr5 = []
#print val[0], type(val[0])
#print val
for i in range(0,len(val5)):
	count_attr_total5 = 0
	count_success5 = 0
	for j in range (0, df.shape[0]): 
		if df.ambition_partner.values[j] == val5[i] :
			count_attr_total5+= 1
			if df.decision.values[j] == 1:
				count_success5+= 1
	count_attr_total_arr5.append(count_attr_total5)
	count_success_arr5.append(count_success5)

#print "loop ", count_attr_total_arr, count_success_arr

success_rate_attractive_arr5 = []

for i in range(0,len(val5)):
	success_rate_attractive_arr5.append(float(count_success_arr5[i])/count_attr_total_arr5[i])
#print success_rate_attractive_arr

# Draw Scatter Plot
x5 = val5
y5 = success_rate_attractive_arr5
colors = (0.1,0.8,.06)
area = np.pi*3
 
# Plot
plt.scatter(x5, y5, s=area, c= 'r', alpha=0.5)
plt.title('Scatter plot of success rate for attribute ambition_partner')
plt.xlabel('ambition_partner')
plt.ylabel('Success rate')
plt.show()

val6 =  df.shared_interests_partner.unique()
success_rate_attractive_arr6= []
count_attr_total_arr6 = []
count_success_arr6 = []
#print val[0], type(val[0])
#print val
for i in range(0,len(val6)):
	count_attr_total6 = 0
	count_success6 = 0
	for j in range (0, df.shape[0]): 
		if df.shared_interests_partner.values[j] == val6[i] :
			count_attr_total6+= 1
			if df.decision.values[j] == 1:
				count_success6+= 1
	count_attr_total_arr6.append(count_attr_total6)
	count_success_arr6.append(count_success6)

#print "loop ", count_attr_total_arr, count_success_arr

success_rate_attractive_arr6 = []

for i in range(0,len(val6)):
	success_rate_attractive_arr6.append(float(count_success_arr6[i])/count_attr_total_arr6[i])
#print success_rate_attractive_arr

# Draw Scatter Plot
x6 = val6
y6 = success_rate_attractive_arr6
colors = (0.1,0.8,.06)
area = np.pi*3
 
# Plot
plt.scatter(x6, y6, s=area, c='r', alpha=0.5)
plt.title('Scatter plot of success rate for attribute shared_interests_partner')
plt.xlabel('shared_interests_partner')
plt.ylabel('Success rate')
plt.show()
'''
######################### data to plot ##################
n_groups = 6
mean_female = (female.attractive_important.mean(), female.sincere_important.mean(), female.intelligence_important.mean(), female.funny_important.mean(), female.ambition_important.mean(), female.shared_interests_important.mean())
mean_male = (male.attractive_important.mean(), male.sincere_important.mean(), male.intelligence_important.mean(), male.funny_important.mean(), male.ambition_important.mean(), male.shared_interests_important.mean())
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, mean_female, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Female')
 
rects2 = plt.bar(index + bar_width, mean_male, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Male')
 
plt.xlabel('gender of participant')
plt.ylabel('preference scores of participant')
plt.title('Scores by participant')
plt.xticks(index + bar_width, ('attractive_important', 'sincere_important', 'intelligence_important', 'funny_important','ambition_important','shared_interests_important'))
plt.legend()
 
plt.tight_layout()
plt.show()
'''

