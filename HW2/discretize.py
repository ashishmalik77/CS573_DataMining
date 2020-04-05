import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt

# importing data
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1')
'''
for i in df:
	print max(df[i].values)
'''	
for i in range(0, df.shape[0]):
	if df.gaming.values[i] > 10:
		df.gaming.values[i] = 10
	if df.reading.values[i] > 10:
		df.reading.values[i] = 10
#print max(df.gaming.values), max(df.reading.values), df.reading.values[1]
'''
for i in df:
	print max(df[i].values)
'''
for i in df:
	#print i	
	if i != 'Unnamed: 0' and i != 'gender' and i != 'race' and i != 'race_o' and i!= 'samerace' and i!='field' and i != 'decision' and i!='attractive_important' and i!='ambition_partner' and i!='shared_interests_partner' and i!='museums' and i!='tv' and i !='interests_correlate':
		#print 'hi'
		df[i] = pd.cut(df[i],5,labels=['0','1','2','3','4'])
		hist = df.groupby(i)[i].count().reset_index(name='Count').to_dict(orient='records')
		val = []
		for j in range(0,len(hist)):
			val.append(hist[j].values()[0])
		print i,": ", val

	if i=='attractive_important' or i=='ambition_partner' or i=='shared_interests_partner' or i=='museums' or i=='tv' or i =='interests_correlate':
		df[i] = pd.cut(df[i],5,labels=['0','1','2','3','4'])
		hist = df.groupby(i)[i].count().reset_index(name='Count').to_dict(orient='records')
		val = []
		for j in range(0,len(hist)):
			val.append(hist[j].values()[1])
		print i,": ", val
df.to_csv(sys.argv[2])



