import numpy as np
#from sklearn.cluster import KMeans
import csv

def preprocessData():
	"""
	This method opens a csv file and preporcess the information
	potential changes:
	1.- Provide it with the name of the file to open
	2.- Provide it with more options, such as the ammount of clusters wanted
	3.- Have it remove the description and resolution columns instead of doing it manually
	5.- Fix the warning
	6.- Change category into format (0,0,....1,0) if possible, or consider pair (like 'THEFT/LARCENY') as
		a new category. This will allow using only one column as category 
	7. More ideas?
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
                if("Monday" in data[i][2]):
                    data[i][2] = 0
                elif("Tuesday" in data[i][2]):
                    data[i][2] = 1
                elif("Wednesday" in data[i][2]):
                    data[i][2] = 2
                elif("Thursday" in data[i][2]):
                    data[i][2] = 3
                elif("Friday" in data[i][2]):
                    data[i][2] = 4
                elif("Saturday" in data[i][2]):
                    data[i][2] = 5
                elif("Sunday" in data[i][2]):
                    data[i][2] = 6

		data[i][4] = labels[i]
		i+=1
	
        
        data = np.array(data)

        if(":" in data[0][0] and "-" in data [0][0]):
            #need to split dates
            times = [dates[0].split(" ")[1] for dates in data]
            dates = [dates[0].split(" ")[0].split("-") for dates in data]
            times = np.array([times])
            dates = np.array(dates)
            data = np.concatenate((data, np.atleast_1d(times.T)), axis=1)
            data = np.concatenate((data, dates), axis = 1)
            data = np.delete(data, 0, 1)

	return data

prepData = preprocessData()
