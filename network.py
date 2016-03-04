import os

def train():
    """
    This method calls the neural network program and trains it.
    The program runs off two special files, called training.txt
    and validation.txt . At the end of training, the program
    prints the confusion matrix to a file called confusion_matrixt.txt
    and the precision/recall of each class to a file called 
    precision_recall.txt. This file also contains, at the end,
    a value of the accuracy of the system over the validation data.

    Finally, these files represent the categories sorted
    ALPHABETICALLY
    """
    os.system("g++ Source.cpp")
    os.system("./a.out A")

def tests():
    """
    This method calls the neural network program to run it
    over a test file. It runs over a test file separated by spaces
    rather than comas. In order to call this method, the network
    must have been trained previously. When the run is over,
    the program prints the results to a csv file that is ready to
    be uploaded to kaggle.
    """
    os.system("g++ Source.cpp")
    os.system("./a.out B")

def predict():
    """
    This method calls the neural network program to run it over
    a file containing the required data from the user. The file
    must be a csv file called "predict.csv" and must be separated
    by spaces. Finally, the result is printed to a file called
    resultsPredict.csv with the categories ordered alphabetically
    and separated by commas. 
    """
    os.system("g++ Source.cpp")
    os.system("./a.out C")


train()
tests()
predict()
