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
from sklearn.metrics import precision_recall_fscore_support
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
    
    
def get_classnames(classfilename='dictionary.txt'):
    
    class_names = []
    with open(classfilename) as classdata:
        for line in classdata:
            class_names.append(line.strip('\n'))
    return class_names
    

#--------------FUNCTIONS FOR TRAINING THE CLASSIFIER----------------------
        
def kNN_train_classifier(train_x, train_y, validation_x, validation_y):
    """ Returns a classifier trained using the training data
    """
    classifier = kNN()
    classifier.fit(train_x, train_y)
    validation_pred = classifier.predict(validation_x)
    
    return validation_pred
    
    
def NB_train_classifier(train_x, train_y, validation_x, validation_y):
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


# Predicting an instance of the data
def predict_instance(classifier, instance_x):
    """ Given a particular instance of data, it returns the class name for that
    instance.NOT WORKING YET
    """
    return classifier.predict(instance_x)
    
    
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
    
    
    
def print_confusion_matrix(validation_y, validation_pred):
    """ Prints the confusion matrix in a tabular form. NOT WORKING YET
    """
    
    val_y = list(validation_y)
    val_pred = list(validation_pred)
    
    tagset = get_classnames()
    
    cfm = confusion_matrix(val_y, val_pred, labels=tagset)
    print '\t',
    for idx in range(len(tagset)):
        print '%s  ' %tagset[idx],
    print '\n', 
    for idx in range(len(cfm)):
        print tagset[idx],
        for iidx in range(len(cfm[idx])):
            print '\t%d' %cfm[idx][iidx],
        print '\n',
        
        
def print_accuracy(validation_y, validation_pred):
    """ Returns the accuracy fo the validation.
    """
    print accuracy_score(validation_y, validation_pred)
    

def print_micro_averages(validation_y, validation_pred):
    """ Prints the micro and macro average precision recall and fscore
    """
    val_y = list(validation_y)
    val_pred = list(validation_pred)
    
    preRecF = precision_recall_fscore_support(val_y, val_pred, average='micro')

    print '\t\t  P\tR\tF' 
    print 'Micro-average:  %0.4f\t%0.4f\t%0.4f' %(preRecF[0], preRecF[1], preRecF[2])
   


        
# Uncomment to test the system
train_x, validation_x, train_y, validation_y = get_data('preprocessed_data.csv')
clf = NB_train_classifier(train_x, train_y, validation_x, validation_y)
validation_pred = getPredictions(clf, validation_x)
print get_classification_report(validation_y, validation_pred)
print
#print getLogLoss(validation_y, validation_pred)
#print
print_accuracy(validation_y, validation_pred)
print
print_micro_averages(validation_y, validation_pred)
print 
predictions = getPredictProbabilities(clf, validation_x)
print getPredictProbabilities(clf, validation_x)
print '\n\n'



# USING THE TEST DATA
print 'Testing it using the test data'
test_x, test_y = get_test_data('test_data.csv')
test_pred = getPredictions(clf, test_x)
print get_classification_report(test_y, test_pred)
print
print_accuracy(test_y, test_pred)
print
print_micro_averages(test_y, test_pred)
print 
predictions = getPredictProbabilities(clf, test_x)
print getPredictProbabilities(clf, test_x)

#writing test results to csv file
result = pd.DataFrame(predictions, columns=get_classnames())
result.to_csv('Result.csv', index = True, index_label = 'Id' )


    
       


