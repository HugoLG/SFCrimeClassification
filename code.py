import numpy as np
from sklearn.cluster import KMeans
import csv

def preprocessData(k):
	"""
	This method opens a csv file and preporcess the information
	Receives:
	integer k: number of clusters with which to separate the addresses
	
	Returns
	np.array Data: The order of the columns is the following
		Category Day(Monday = 0) PdDistrict AddressCluster Time Year Month DayNumber
	dictionary Category: Contains the mapping of the categories to numbers
	dictionary District: Contains the mapping of the districts to numbers
	
	potential changes:
	1.- Provide it with the name of the file to open
	3.- Have it remove the description and resolution columns instead of doing it manually
	5.- Fix the warning
	7. More ideas?
	"""
	with open('train.csv','r') as dest_f:
		data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
		data = [data for data in data_iter]
	#Gets rid of the header, could also be changed to receive a file with no headers
	data = data[1:]
	clusterData = [clusterData[5:7] for clusterData in data]

        clusterData = np.array(clusterData)
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(clusterData)
	labels = kmeans.predict(clusterData)

        counterCategory = 0
        counterPdDistrict = 0
        categoryDict = {}
        pdDistrictDict = {}
        
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

                if not categoryDict.has_key(data[i][1]):
                    categoryDict[data[i][1]] = counterCategory
                    counterCategory += 1
                data[i][1] = categoryDict[data[i][1]]

                if not pdDistrictDict.has_key(data[i][3]):
                    pdDistrictDict[data[i][3]] = counterPdDistrict
                    counterPdDistrict += 1
                data[i][3] = pdDistrictDict[data[i][3]]
                #substitute the addresses for cluster labels
		data[i][4] = labels[i]
		i+=1
	
        data = np.delete(data, 5, 1);
        data = np.delete(data, 5, 1);
        data = np.array(data)

        if(":" in data[0][0] and "-" in data [0][0]):
            #need to split dates
            times = [(int(dates[0].split(" ")[1].split(":")[0])*60+int(dates[0].split(" ")[1].split(":")[1])) for dates in data]
            dates = [dates[0].split(" ")[0].split("-") for dates in data]
            times = np.array([times])
            dates = np.array(dates)
            data = np.concatenate((data, np.atleast_1d(times.T)), axis=1)
            data = np.concatenate((data, dates), axis = 1)
            data = np.delete(data, 0, 1)

	return data, categoryDict, pdDistrictDict

prepData, catDict, districtDict = preprocessData(15)

print prepData[0]
#print str(catDict)
#print str(districtDict)
dataset = prepData

vectorizer = CountVectorizer()
#dataset = dataset.reshape(-1,7)
#print dataset

data = dataset[:,1]
#print data
for i in range(2,len(dataset[0])):
    data =np.column_stack((data,dataset[:,i]))
#print "dataset only"
#print data
DictList = []

def createDic(DictList,raw):
    #each column
    newList = [0 for i in range(len(raw[0]))]
    for i in range(len(raw[0])):
        dic = dict()
        c = 0
        for j in range(len(raw)):
            if raw[j][i] not in dic:
                dic[raw[j][i]] = c
                c += 1
        DictList.append(dic)
        #print dic
        Y = [0 for index in range(len(raw))]
        
        for k in range(len(raw)):
            #print k, i
            #print raw[k][i]
            Y[k] = dic[raw[k][i]]
        if i == 0:
            newList = Y
        else:
            newList = np.column_stack((newList,Y))
    #print newList
    return newList

data = createDic(DictList,data)
target = dataset[:,0]
print("%d documents" % len(data))

print

dic = dict()
countCat = 0
for i in range(len(target)):
    if target[i] not in dic:
        dic[target[i]] = countCat
        countCat += 1
X = data#vectorizer.fit_transform(data)
#print ("length X %d" %len(X))
print X
Y = [0 for i in range(len(target))]
#print ("length Y %d" %len(Y))
for i in range(len(target)):
    Y[i] = dic[target[i]]
#print target
#print Y
print("%d categories" % len(catDict))

def QuickReduct(C,D): 
    gammaCD = 0
    for t in D:
        for num in t:
            gammaCD = gammaCD+num
    gammaRD = 0
    T = []
    R = []
    
    while gammaRD < gammaCD:
        T = R
        X = list(set(C) - set(R))
        for index in range(len(X)):
            gammaRXD = gammaRD
            for num in range(len(D)):
                gammaRXD = gammaRXD+D[num][index]
            if(gammaRXD > gammaRD):
                R.append(X[index])
                T = R
                gammaRD = gammaRXD
            R = T
    return R

def trainTrees(numTrees, Xt, yt):
    X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.500000, random_state=42)

    half = len(y_train)
    sampleSize = (int)(half*0.8) 
    """can be changed"""

    
    DT = []
    """create pool of decision trees"""
    for count in range(numTrees):
        tr,tt,te,TT = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
        dt = tree.DecisionTreeClassifier()
        
        dt.fit(tr,te)
        DT.append(dt)

    DecisionTable = [[0 for x in range(numTrees)]for x in range(half)]

    for index in range(numTrees):
        for doc in range(half):
            dt = DT[index]
           
            b = dt.predict(X_train[doc])
            DecisionTable[doc][index] = 1 if y_train[doc] == b else 0

    slt = QuickReduct(DT,DecisionTable)
    return slt
    
times = 0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.100000, random_state=42)
pool = trainTrees(11,X_train,y_train)

confTable = [[[0 for x in range(2)]for x in range(2)]for x in range(countCat)]
    
for index in range(len(y_test)):
    countResult = [0 for x in range(len(catDict))]
    for t in pool:
        result = t.predict(X_test[index])
        countResult[result] = countResult[result]+1  
    maxIndex = countResult.index(max(countResult))
    ind = y_test[index]    
    if(y_test[index] == maxIndex):
        confTable[ind][0][0] +=1 #TP
        for i in range(countCat):
            if i != ind:
                confTable[i][1][1] += 1 #TN of else
    else:
        confTable[ind][0][1] += 1 #FP of index
        confTable[maxIndex][1][0] += 1 #FN of maxIndex

print confTable
TP,TN,FN,FP = 0,0,0,0
acc_l, pre_l, rec_l = 0.0,0.0,0.0
    #TP_l, TN_l, FN_l, FP_l = 0,0,0,0
for index in range(len(confTable)):
    TP_l = confTable[index][0][0]
    TN_l = confTable[index][1][1]
    FN_l = confTable[index][1][0]
    FP_l = confTable[index][0][1]
        
    if TP_l != 0 | FN_l !=0 | FP_l != 0:
        times += 1
    
        TP = TP+TP_l
        TN = TN+TN_l
        FN = FN+FN_l
        FP = FP+FP_l
        #print TP, FP, FN, TN
    #print correctly, len(y_test)
print times
TP = (float)(TP)/(float)(times)
FP = (float)(FP)/(float)(times)
FN = (float)(FN)/(float)(times)
TN = (float)(TN)/(float)(times)
    
precision = TP/(TP+FP)
recall = TP/(TP+FN)
    
print ("precision %f" %(precision))
print ("recall %f" %(recall))
    

print ("accuracy %f" %((TP+TN)/(TP+FP+FN+TN)))
