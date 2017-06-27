# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:28:40 2017

@author: DT
"""
import numpy as np

# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'pelease'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cote', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

# 创建词汇表向量
def createVocabList(dataSet):
    vocabSet = set([])                                  #创建空集
    for document in dataSet:                            #创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 输出样本词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                    #创建一个所有元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:   
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)                        #样本数 
    numWords = len(trainMatrix[0])                         #词汇数
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)                             #初始化概率
    p1Num = np.ones(numWords)
    p0Denm = 2.0
    p1Denm = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                           #向量相加
            p1Num += trainMatrix[i]
            p1Denm += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denm += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denm)
    p0Vect = np.log(p0Num/p0Denm)
    return p0Vect, p1Vect, pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #元素相乘
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V,p1V,pAb))

#朴素贝叶斯词袋模型    
def bagOfwords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 文件解析及完整的垃圾邮件测试函数
def textParse(bigString):                                               #解析字符串
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) >2]

def spamTest():                                                        #对贝叶斯垃圾邮件分类器进行自动化处理    
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):                                             #导入并解析文本
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in list(range(10)):                                                 #随机构建训练集
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:                                            #对测试集分类
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error: ", docList[docIndex])
    print('The error rate is: ', float(errorCount)/len(testSet))

#RSS源分类器及高频词去除函数    
def calcMostFreq(vocabList, fullText):                                  #计算出现频率
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):                                             #每次访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])                                  #去掉出现次数最高的那些词
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfwords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfwords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1 
    print('The error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0v, p1v
    
def getTopWords(ny, sf):
    import operator
    vocabList,p0v,p1v = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            topSF.append((vocabList[i],p0v[i]))
        if p1v[i] > -6.0:
            topNY.append((vocabList[i],p1v[i]))
    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse = True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key = lambda pair: pair[1], reverse = True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
        
    
    
