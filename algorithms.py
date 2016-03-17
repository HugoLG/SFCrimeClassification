# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:27:49 2016

@author: alexa
"""

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd

#-----------FUNCTIONS FOR GETTING DATA INTO PYTHON -----------------------
def get_data(filename):
    """ Returns the data needed for training and validation
    """

    data = pd.read_csv(filename)

    # the headings of the features
    headers = data.columns.values

    # to hold all the features
    features = np.delete(headers,-1)

    # the label heading
    label_header = headers[-1]

    # training data
    train_data = data[features]

    # training target
    target = data[label_header]

    # Getting the training and validation
    train_x, validation_x, train_y, validation_y = \
    train_test_split(train_data, target, test_size=0.30, random_state=42)

    return train_x, validation_x, train_y, validation_y

def get_test_data(filename):
    """ Returns the data needed for training and validation
    """

    data = pd.read_csv(filename)

    # the headings of the features
    headers = data.columns.values

    # training data
    test_data = data[headers]

    # Getting the training and validation

    return test_data


def get_classnames(classfilename='dictionary.txt'):

    class_names = []
    with open(classfilename) as classdata:
        for line in classdata:
            class_names.append(line.strip('\n'))
    return class_names


#--------------FUNCTIONS FOR TRAINING THE CLASSIFIER--------------------------
def NB_train_classifier(train_x, train_y):
    """ Returns the predictions on the validation set
    """
    classifier = BernoulliNB()
    classifier.fit(train_x, train_y)

    return classifier

#---------------FUNCTIONS FOR MAKING PREDICTIONS--------------------------
def getPredictions(classifier, validation_x):
    """Returns the predictions for a given classifier using
    the validation data.
    """
    return classifier.predict(validation_x)

def getPredictProbabilities(classifier, validation_x):
    """Get the predicted probabilities
    """
    return classifier.predict_proba(validation_x)


def predict_instance(classifier, instance_x):
    """ Given a particular instance of data, it returns the class name for that
    instance.
    """
    numCat = len(get_classnames())
    y = []
    for index in range(len(instance_x)):
        countResult = [0 for x in range(numCat)]
        try:
            result = classifier.predict(instance_x.iloc[index])
            countResult[result] = countResult[result]+1
        except ValueError:
            pass
        maxIndex = countResult.index(max(countResult))
        y.append(maxIndex)
    return y

#-------------- FUNCTIONS FOR EVALUATING THE CLASSIFIER-------------------
def getLogLoss(validation_y, validation_pred):
    """Returns the log loss
    """
    return log_loss(validation_y, validation_pred)


def get_classification_report(validation_y, validation_pred):
    """ Returns the classification report for the given classify.
    It uses predicts the labels of the validation set and uses
    that as a bases for testing the perfomance of the classifier
    """

    lb = LabelBinarizer()
    val_y = lb.fit_transform(list(validation_y))
    val_pred = lb.transform(list(validation_pred))

    tagset = get_classnames()

    return classification_report(val_y,val_pred,target_names=tagset)


def print_accuracy(validation_y, validation_pred):
    """ Returns the accuracy fo the validation.
    """
    print accuracy_score(validation_y, validation_pred)



#------Putting it together-----------------------------------------------------
def NB_clf_system(filename):
    """A function that trains a Bernoulli Naive Bayes classifier and returns
    the precision, recall and accuracy for the trained classifier.
    """
    train_x, validation_x, train_y, validation_y = get_data(filename)

    clf = NB_train_classifier(train_x, train_y)

    validation_pred = getPredictions(clf, validation_x)

    precision = precision_score(validation_y, validation_pred, average='micro')

    recall = recall_score(validation_y, validation_pred, average='micro')

    accuracy = accuracy_score(validation_y, validation_pred)

    return precision, recall, accuracy, clf


#----------Getting the confusion matrix (adapted from Mike)------------------
def get_confusion_matrix(filename):
    """ Confusion Matrix for each category
    """
    prepData = pd.read_csv(filename)#,sep = ' ')

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
    clf = NB_train_classifier(X_train,y_train)

    confTable = [[[0 for x in range(2)]for x in range(2)]for x in range(countCat)]

    correctly = 0.0

    for index in range(len(y_test)):
        countResult = [0 for x in range(countCat)]
        result = clf.predict(X_test.iloc[index])
        countResult[result] = countResult[result]+1
        maxIndex = countResult.index(max(countResult))


        yti = y_test.iloc[index]

        if(yti == maxIndex):
            correctly = correctly + 1
            confTable[yti][0][0] = confTable[yti][0][0]+1
            for i in range(countCat):
                if i != yti:
                    confTable[i][1][1] = confTable[i][1][1]+1
        else:
            confTable[yti][0][1] = confTable[yti][0][1]+1
            confTable[result][1][0] = confTable[result][1][0]+1


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
    return confTable


def printConfusionTable(confTable):
    """Print confusion table of this training
    """
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
#
#
#
#
#
##writing test results to csv file
#train_x, validation_x, train_y, validation_y = get_data('preprocessed_data.csv')
#clf = NB_train_classifier(train_x, train_y)
#
#test_data = get_test_data('preprocessed_testing.csv')
#testPredictions = getPredictProbabilities(clf, test_data)
#
#result = pd.DataFrame(testPredictions, columns=get_classnames())
#result.to_csv('Result.csv', index = True, index_label = 'Id' )
#
#
## Printing the confusion matrix









