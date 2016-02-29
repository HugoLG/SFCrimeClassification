import pandas as pd
import re
import math
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
 

def preprocess(file):
    """
    Receives the name of the file that is used for obtaining the data.
    Returns
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
        Label Encoder: contains the information of the categories
    Refer to http://efavdb.com/predicting-san-francisco-crimes/     in order to know
    how to use data frames in scikit learn code. For example:
    features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
    'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
    'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
 
    training, validation = train_test_split(train_data, train_size=.60)
    model = BernoulliNB()
    model.fit(training[features], training['crime'])
    The method is still missing the preprocessing of the address, ill finish that later
    """

    train=pd.read_csv(file, parse_dates = ['Dates'])

    categoryDict = {}
    counterCategory = 0

    #Convert crime labels to numbers
    crime = train.Category


    for i in range(0, len(crime)):
        if not categoryDict.has_key(crime[i]):
            categoryDict[crime[i]] = counterCategory
            counterCategory += 1

    crime_processed = [categoryDict[c] for c in crime] 
    crime_processed = pd.DataFrame(crime_processed)


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

    #work on address
    """
    Ill work on this later...
    address = train.Address
    rgx = re.compile('( AL( |\Z|\s))|( ALY( |\Z|\s))|( ARC( |\Z|\s))|( AV( |\Z|\s))|( AVE( |\Z|\s))|( BL( |\Z|\s))|( BLVD( |\Z|\s))|( BR( |\Z|\s))|( BYP( |\Z|\s))|( CSWY( |\Z|\s))|( CR( |\Z|\s))|( CTR( |\Z|\s))|( CIR( |\Z|\s))|( CT( |\Z|\s))|( CRES( |\Z|\s))|( DR( |\Z|\s))|( EXPY( |\Z|\s))|( EXT( |\Z|\s))|( FWY( |\Z|\s))|( GDNS( |\Z|\s))|( GRV( |\Z|\s))|( HTS( |\Z|\s))|( HWY( |\Z|\s))|( HY( |\Z|\s))|( LN( |\Z|\s))|( MNR( |\Z|\s))|( PARK( |\Z|\s))|( PL( |\Z|\s))|( PZ( |\Z|\s))|( PLZ( |\Z|\s))|( PT( |\Z|\s))|( RD( |\Z|\s))|( RW( |\Z|\s))|( RTE( |\Z|\s))|( R( |\Z|\s))|( SQ( |\Z|\s))|( ST( |\Z|\s))|( TER( |\Z|\s))|( TR( |\Z|\s))|( TRL( |\Z|\s))|( TPKE( |\Z|\s))|( VIA( |\Z|\s))|( VIS( |\Z|\s))|( WAY( |\Z|\s))|( WY( |\Z|\s))|( WK( |\Z|\s))|( I-80( |\Z|\s))|( VIA( |\Z|\s))|( MAR( |\Z|\s))')
    f = lambda x: rgx.search(x).group()
    address = address.map(f)
    print address
    """

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
    train_data = pd.concat([hour, day, month, year, days, district, X, Y], axis=1)
    train_data['crime']=crime_processed
    
    train_data.to_csv("preprocessed_data.csv", sep=',')

    out_file =open('dictionary.txt', "w")
    for k in sorted(categoryDict, key=categoryDict.get):
        out_file.write(k)
        out_file.write("\n")

    out_file.close

    return train_data


preprocess('train.csv')

