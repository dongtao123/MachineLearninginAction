# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:49:51 2017

@author: DT
"""

import matplotlib.pyplot as plt

# 使用额我那本注解绘制树节点
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")          #定义文本框和箭头格式
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',\
                            xytext = centerPt, textcoords = 'axes fraction',\
                            va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)
    
def createPlot():
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plt.rc('font', family='SimHei', size=13)
    plotNode('决策节点', (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
    
# 获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():                               #测试节点的数据类型是否为字典
        if type(secondDict[key])._name_=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.key():
        if type(secondDict[key])._name_=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth:    maxDepth = thisDepth
    return maxDepth


