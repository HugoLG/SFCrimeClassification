# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:27:49 2016

@author: alexa
"""

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd


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
    validation_pred = classifier.predict(validation_x)
    
    return validation_pred
    
def get_classification_report(validation_y, validation_pred, dictfile):
    """ Returns the classification report for the given classify.
    It uses predicts the labels of the validation set and uses 
    that as a bases for testing the perfomance of the classifier
    """
    #validation_pred = classifier.predict(validation_x)
    
    lb = LabelBinarizer()
    val_y = lb.fit_transform(list(validation_y))
    val_pred = lb.transform(list(validation_pred))
        
    #tagset = [str(n) for n in set(lb.classes_)]
    tagset = get_classnames()

    return classification_report(val_y,val_pred,target_names=tagset)
    
    
def print_confusion_matrix(validation_y, validation_pred, dictfile):
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
   
        
def predict_instance(classifier, instance_x):
    """ Given a particular instance of data, it returns the class name for that
    instance.NOT WORKING YET
    """
    return classifier.predict(instance_x)
       


