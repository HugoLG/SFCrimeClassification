import pandas as pd
import re
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np


def preprocess(file, isTraining, isNeuralNetwork):
    """
    Receives the name of the file that is used for obtaining the data and a boolean.
    If the boolean is true, the preprocess prepares the training file
    If the boolean is false, the preprocess prepares the testing file
    Returns
        Nothing, prints a file with the following format:
	DataFrame Data: The order of the columns is the following
            Two columns containing the information about the hour using sine and cosine
            Two columns containing the information about the day number using sine and cosine
            Two columns containint the information about the month number using sine and cosine
            N (12 i think) columns containing a  binarized string representation of the year
            7 columns containing a binarized string representation of the day of the week
            N (9 i think) columns containing a binarized string representation of the districts
            One column with the normalized X coordinate
            One column with the normalized Y coordinate
            One column with labels representing the diff. crime categories
        Prints a dictionary for categories

    The method is still missing the preprocessing of the address, ill finish that later
    """

    train=pd.read_csv(file, parse_dates = ['Dates'])

    categoryDict = {}
    counterCategory = 0

    #Convert crime labels to numbers
    if isTraining:
        crime = train.Category


        for i in range(0, len(crime)):
            if not categoryDict.has_key(crime[i]):
                categoryDict[crime[i]] = counterCategory
                counterCategory += 1

        crime_processed = [categoryDict[c] for c in crime]
        crime_processed = pd.DataFrame(crime_processed)

    if isNeuralNetwork:
        crime2 = pd.get_dummies(train.Category)

    #Get binarized weekdays and districts.
    days = pd.get_dummies(train.DayOfWeek)
    days = days.reindex_axis(sorted(days.columns), axis = 1)
    district = pd.get_dummies(train.PdDistrict)
    district = district.reindex_axis(sorted(district.columns), axis = 1)


    #convert hours to the circle format
    hour = train.Dates.dt.hour
    col1 = [int(math.sin(2*h*math.pi/24.0)*1000)/1000.0 for h in hour]
    col2 = [int(math.cos(2*h*math.pi/24.0)*1000)/1000.0 for h in hour]
    hour = pd.concat([pd.DataFrame(col1), pd.DataFrame(col2)], axis = 1)

    #convert months to the circle format
    month = train.Dates.dt.month
    col1 = [int(math.sin(2*h*math.pi/12.0)*1000)/1000.0 for h in month]
    col2 = [int(math.cos(2*h*math.pi/12.0)*1000)/1000.0 for h in month]
    month = pd.concat([pd.DataFrame(col1), pd.DataFrame(col2)], axis = 1)

    #convert days to the circle format
    day = train.Dates.dt.day
    col1 = [int(math.sin(2*h*math.pi/31.0)*1000)/1000.0 for h in day]
    col2 = [int(math.cos(2*h*math.pi/31.0)*1000)/1000.0 for h in day]
    day = pd.concat([pd.DataFrame(col1), pd.DataFrame(col2)], axis = 1)

    #binarize years
    year = train.Dates.dt.year
    year = pd.get_dummies(year)
    year = year.reindex_axis(sorted(year.columns), axis=1)

    #work on X and Y
    X = train.X
    #normalize X
    preprocessing.normalize(X, norm = 'max', copy = False)
    #reduce decimal places
    f = lambda x: '%.5f' % float(x)
    X = X.map(f)
    X = pd.DataFrame(X)

    Y = train.Y
    #normalize Y
    preprocessing.normalize(Y, norm = 'max', copy = False)
    #reduce decimal places
    f = lambda x: '%.5f' % float(x)
    Y = Y.map(f)
    Y = pd.DataFrame(Y)


    #Build new array
    if isTraining:
        if isNeuralNetwork:
            train_data = pd.concat([hour, day, month, year, days, district, X, Y, crime2], axis=1)
        else:
            train_data = pd.concat([hour, day, month, year, days, district, X, Y], axis=1)
            train_data['crime']=crime_processed
        out_file =open('dictionary.txt', "w")
        for k in sorted(categoryDict, key=categoryDict.get):
            out_file.write(k)
            out_file.write("\n")

        out_file.close
        train_data.to_csv("preprocessed_data.csv", sep=',')

    else:
        train_data.to_csv("preprocessed_testing.csv", sep=',')

    return train_data

if __name__ == '__main__':
    preprocess('singlePredictionData.csv', False, True)
