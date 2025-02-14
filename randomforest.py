import numpy as np
from sklearn.cluster import KMeans
import csv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import shuffle
import numpy
import random
import pandas as pd
import postprocessing
import sys
import warnings
warnings.filterwarnings("ignore")


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
    #can be changed


    DT = []
    #create pool of decision trees
    for count in range(numTrees):
        tr,tt,te,TT = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
        dt = tree.DecisionTreeClassifier()

        dt.fit(tr,te)
        DT.append(dt)

    DecisionTable = [[0 for x in range(numTrees)]for x in range(half)]

    for index in range(numTrees):
        for doc in range(half):
            dt = DT[index]

            b = dt.predict(X_train.iloc[doc])
            DecisionTable[doc][index] = 1 if y_train.iloc[doc] == b else 0

    slt = QuickReduct(DT,DecisionTable)
    return slt


def randomForest(file):
    prepData = pd.read_csv(file)#,sep = ' ')

    print 'finished read data'
    headers = prepData.columns.values
    #print("heaaders....")
    #print(headers)

    features = np.delete(headers,-1)
    #print type(features)
    #features = np.delete(headers,0)
    #print("headers -1 ...")
    #print(len(headers))
    targetH = headers[-1]
    #print(targetH)

    data = prepData[features]
    target = prepData[targetH]
    #print(target)
    countCat = max(target)+1

    #print data
    #print target

    print 'start training'
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.200000, random_state=42)
    pool = trainTrees(3,X_train,y_train)
    #print len(pool)
    print 'finished training'

    correctly = 0.0

    """create confusion table of this training"""
    confTable = [[[0 for x in range(2)]for x in range(2)]for x in range(countCat)]
    for index in range(len(y_test)):
        countResult = [0 for x in range(countCat)]
        for t in pool:
            result = t.predict(X_test.iloc[index])
            countResult[result] = countResult[result]+1
        maxIndex = countResult.index(max(countResult))
        #print type(y_test)
        yti = y_test.iloc[index]
        #print yti, maxIndex
        if(yti == maxIndex):
            correctly = correctly+1
            confTable[yti][0][0] = confTable[yti][0][0]+1 #TP
            for i in range(countCat):
                if i != yti:
                    confTable[i][1][1] = confTable[i][1][1]+1 #TN of else
        else:
            confTable[yti][0][1] = confTable[yti][0][1]+1 #FP of index
            confTable[maxIndex][1][0] = confTable[maxIndex][1][0]+1 #FN of maxIndex


    """start computing accuracy, precision, recall"""
    TP,TN,FN,FP = 0,0,0,0
    print 'finished'
    for index in range(countCat):
        #print index
        TP = TP+confTable[index][0][0]
        TN = TN+confTable[index][1][1]
        FN = FN+confTable[index][1][0]
        FP = FP+confTable[index][0][1]
        #print TP, FP, FN, TN

    TP = (float)(TP)/(float)(len(data))
    FP = (float)(FP)/(float)(len(data))
    FN = (float)(FN)/(float)(len(data))
    TN = (float)(TN)/(float)(len(data))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print ("precision %f" %(precision))
    print ("recall %f" %(recall))
    print ("correctly classified %f" %(correctly/len(y_test)))

    #this is the accuracy by definition, include the TN into computation
    print ("accuracy %f" %((TP+TN)/(TP+FP+FN+TN)))
    return pool,confTable,precision, recall, (correctly/len(y_test))

"""print confusion table of this training"""
def printConfusionTable(confTable):
    for i in range(len(confTable)):
        print i
        try:
            precision = (float)(confTable[i][0][0])/(float)(confTable[i][0][0]+confTable[i][0][1])
        except ZeroDivisionError:
            precision = 'not measuarable'
        try:
            recall = (float)(confTable[i][0][0])/(float)(confTable[i][0][0]+confTable[i][1][0])
        except ZeroDivisionError:
            recall = 'not measurable'

        print '['+str(confTable[i][0][0])+'|'+str(confTable[i][0][1])+']'
        print '['+str(confTable[i][1][0])+'|'+str(confTable[i][1][1])+']'
        print "precision ", precision
        print "recall ", recall

"""predicted another instance"""
def randomForestPredicted(X,numCat,pool):
    y = []
    for index in range(len(X)):
        countResult = [0 for x in range(numCat)]
        for t in pool:
            try:
                result = t.predict(X.iloc[index])
                countResult[result] = countResult[result]+1
            except ValueError:
                pass
        maxIndex = countResult.index(max(countResult))
        y.append(maxIndex)
    return y



if __name__ == '__main__':
    p, ct = randomForest("preprocessed_data.csv")
    printConfusionTable(ct)
    testData = pd.read_csv('preprocessed_testing.csv')#_cut.csv')
    #print testData
    #data = pd.np.array(testData)

    #print data
    #data =  np.delete(data,-1,axis=0)
    y = randomForestPredicted(testData, 39, p)
    #print y

    from postprocessing import writeOutputToCSV
    writeOutputToCSV('run_10trees.csv',y,'dictionary.txt')


#printconfusiontablentConfusionTable()
#print randomForestPredicted([X_test.iloc[0],X_test.iloc[2]])
#print y_test.iloc[0], y_test.iloc[2]
