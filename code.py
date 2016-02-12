import numpy as np
from sklearn.cluster import KMeans
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings("ignore")

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
	with open('D:\Essex\CE903 Group Project\Data\\train_cut.csv','r') as dest_f:
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
dataset = prepData

vectorizer = CountVectorizer()
dataset = dataset.reshape(-1,7)
#print dataset

data = dataset[:,0]
#print data
for i in range(2,len(dataset[0])):
    data =np.column_stack((data,dataset[:,i]))
#print "dataset only"
print data
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
target = dataset[:,1]
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
print("%d categories" % countCat)

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
    
avgacc = 0.0
times = 0
avgpre = 0.0
avgrec = 0.0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.500000, random_state=42)
pool = trainTrees(11,X_train,y_train)
correctly = 0.0

confTable = [[[0 for x in range(2)]for x in range(2)]for x in range(countCat)]
    
for index in range(len(y_test)):
    countResult = [0 for x in range(countCat)]
    for t in pool:
        result = t.predict(X_test[index])
        countResult[result] = countResult[result]+1  
    maxIndex = countResult.index(max(countResult))
    ind = y_test[index]    
    if(y_test[index] == maxIndex):
        correctly = correctly+1
        ind = y_test[index]
        confTable[ind][0][0] = confTable[ind][0][0]+1 #TP
        for i in range(countCat):
            if i != ind:
                confTable[i][1][1] = confTable[i][1][1]+1 #TN of else
    else:
        confTable[ind][0][1] = confTable[ind][0][1]+1 #FP of index
        confTable[maxIndex][1][0] = confTable[maxIndex][1][0]+1 #FN of maxIndex

    #print confTable
TP,TN,FN,FP = 0,0,0,0

for index in range(len(confTable)):
    TP = TP+confTable[index][0][0]
    TN = TN+confTable[index][1][1]
    FN = FN+confTable[index][1][0]
    FP = FP+confTable[index][0][1]
        #print TP, FP, FN, TN
        
TP = (float)(TP)/(float)(len(X))
FP = (float)(FP)/(float)(len(X))
FN = (float)(FN)/(float)(len(X))
TN = (float)(TN)/(float)(len(X))
    
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print ("precision %f" %(precision))
print ("recall %f" %(recall))
    
avgpre = avgpre+precision
avgrec = avgrec+recall
avgacc = avgacc + correctly/len(y_test)
times = times+1
print ("accuracy %f" %(correctly/len(y_test)))
"""
avg = avgacc/times
avgrec = avgrec/times
avgpre = avgpre/times

print ("baseline acc avg: %f" % avg)
print ("baseline precision avg: %f" %avgpre)
print ("baseline recall: %f" %avgrec)
  
clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    sc = clf.score(X_test,y_test)
    print ("test %f" %sc)
    print
"""
    
