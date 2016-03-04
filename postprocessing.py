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
    v.insert(0,'id')
    #print v
    output.append(v)
    count = 0
    for t in y:
        body = []
        body.append(count)
        for i in range(1,len(v)):
            #print t, catDict[v[i]]
            if t == catDict[v[i]]:
                #print t, v[i], catDict[v[i]]
                body.append(1)
                #print body
            else: body.append(0)
        #print 'body', body
        count+=1
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
