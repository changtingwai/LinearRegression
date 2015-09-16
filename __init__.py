# -*- coding: utf-8 -*-
from sklearn import linear_model
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(filename):
    fr=open(filename)
    numFeat=len(fr.readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1]))
    return dataMat,labelMat
data,label=loadDataSet('abalone.txt')

trainData=data[0:99]
# print (trainData)
trainLabel=label[0:99]
testData=data[100:200]
# print len(testData)
testLabel=label[100:200]


clf=linear_model.LinearRegression()
clf.fit(trainData, trainLabel)
result=[]
for i in range(100):
    result.append(int(clf.predict(testData[i])))
errorCount=0
for j in range(100):
    if result[j]!=testLabel[j]:
        errorCount+=1
print errorCount/float(len(testLabel))
print result
print testLabel



plt.show()