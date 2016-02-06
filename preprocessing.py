import numpy as np
from sklearn.cluster import KMeans
import csv

def preprocessData():
	"""
	This method opens a csv file and preporcess the information
	potential changes:
	1.- Provide it with the name of the file to open
	2.- Provide it with more options, such as the ammount of clusters wanted
	3.- Have it remove the description and resolution columns instead of doing it manually
	4.- Separate the timestamp into different fields (year, month, day of the month, time)
	5.- Fix the warning
	6.- More ideas?
	"""
	with open('train.csv','r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
		data = [data for data in data_iter]
	#Gets rid of the header, could also be changed to receive a file with no headers
	data = data[1:]
	clusterData = [clusterData[5:7] for clusterData in data]
	k=15
	clusterData = np.array(clusterData)
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(clusterData)
	labels = kmeans.predict(clusterData)

	#substitute the addresses for cluster labels
	for i in range(0, len(data)):
		data[i][4] = labels[i]
		i+=1
	data = np.array(data)

	return data

prepData = preprocessData()

