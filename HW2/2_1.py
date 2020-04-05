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

###################### 2. (i). ##################################

############## Divide the dataset into two sub-datasets by the gender of participant ################

female = df[df["gender"] == 0]
male = df[df["gender"] == 1]

#print female.shape[0]
#print male.shape[0]

############################### Mean  ################################
print "Mean values for six attributes in female dataset: ",female.attractive_important.mean(), female.sincere_important.mean(), female.intelligence_important.mean(), female.funny_important.mean(), female.ambition_important.mean(), female.shared_interests_important.mean()
print "Mean values for six attributes in male dataset: ",male.attractive_important.mean(), male.sincere_important.mean(), male.intelligence_important.mean(), male.funny_important.mean(), male.ambition_important.mean(), male.shared_interests_important.mean()


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


