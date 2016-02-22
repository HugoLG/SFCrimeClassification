# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:09:05 2016

@author: alexa
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



# Reading the data into python
train = pd.read_csv('train.csv', parse_dates = ['Dates'])
test = pd.read_csv('test.csv', parse_dates = ['Dates'])

# Getting the category of the crime
crime_labels = preprocessing.LabelEncoder()
crime_category = crime_labels.fit_transform(train.Category)

#Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour) 

 
#Build new array
train_data = pd.concat([hour, days, district], axis=1)
train_data['crime_category']= crime_category

#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour) 
 
# Array for the test data
test_data = pd.concat([hour, days, district], axis=1)
 
training, validation = train_test_split(train_data, train_size=.66)


# BUILDING THE CLASSIFIER
daysfeatures = [x for x in days]
districtfeatures = [x for x in district]
hoursf = [h for h in hour]
features = daysfeatures + districtfeatures + hoursf
classifier = kNN()
classifier.fit(training[features], training['crime_category'])
validation_predict = classifier.predict(validation[features])

# Testing the model
print accuracy_score(validation['crime_category'], validation_predict)