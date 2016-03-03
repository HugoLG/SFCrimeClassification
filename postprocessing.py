import sys
import re
import csv

def readDict(filename):
    f = open(filename)
    a = f.readlines()
    dic = dict()
    for i in range(len(a)):
        text = re.sub('\n','',a[i])
        dic[text] = i
    return dic

def createOutput(y,filename):
    
    output = []
    catDict = readDict(filename)
    v = catDict.keys()
    v.sort()
    #print v
    output.append(v)
    for t in y:
        body = []
        for i in range(len(catDict)):
            #print t, catDict[v[i]]
            if t == catDict[v[i]]:
                #print 'e'
                body.append(1)
                #print body
            else: body.append(0)
        #print 'body', body
        output.append(body)
    #print output
    return output
        
#t = createOutput([0,3,5])
#print t
        
def writeOutputToCSV(nameOfFile,y,dic):
    output = createOutput(y,dic)
    resultFile = open(nameOfFile,'wb')
    wr = csv.writer(resultFile,dialect = 'excel')
    wr.writerows(output)


#writeOutputToCSV('test.csv',[0,1,3],'dictionary.txt')
