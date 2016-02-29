import sys
import re
import csv

def createOutput(y,catDict):
    output = []
    v = catDict.values()
    #print v
    output.append(v)
    for t in y:
        body = []
        for i in range(len(catDict)):
            if t == i:
                body.append(1)
            else: body.append(0)
        output.append(body)
    return output
        
#t = createOutput([0,3,5])
#print t
        
def writeOutputToCSV(nameOfFile,y):
    output = createOutput(y)
    resultFile = open('D:\Essex\CE903 Group Project\GIT\GIT\\'+nameOfFile,'wb')
    wr = csv.writer(resultFile,dialect = 'excel')
    wr.writerows(output)

def convertToOutput(y,dic):
    filename = y#'D:\Essex\CE903 Group Project\GIT\GIT\Category.txt'
    f = open(filename)
    a = f.readlines()
    catDict = dict()
    for i in range(len(a)):
        text = re.sub('\n','',a[i])
        catDict[i] = text
    return createOutput(y,dic)

def convertToOutputAndWriteFile(y,dic,filename):
    t = createOutput(y,dic)
    writeOutputToCSV(filename,t)

#writeOutputToCSV('test.csv',[0,1,3])