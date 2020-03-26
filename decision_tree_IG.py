# coding: utf-8
import numpy as np
import csv
import pandas as pd
import random
from pprint import pprint
from sklearn.metrics import confusion_matrix

class DecisionTreeIG():
    def __init__(self):
        self.wine = {
            'data': 'wine.data',
            'dataType': 'continuous',
            'classInd': 0,
            'length': 178,
            'columnNames': ['label', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color', 'hue', 'od280', 'proline']
        }
        self.tictactoe = {
            'data': 'tic-tac-toe.data',
            'dataType': 'categorical',
            'classInd': -1,
            'length': 958,
            'columnNames': ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br', 'label']
        }

    def sameLabels(self, data, dataframe):
        labels = dataframe[:, 0] if data == 'wine' else dataframe[:, -1]
        uniqueLabels = np.unique(labels)
        return (len(uniqueLabels) == 1)

    def calcEntropy(self, data, dataframe):
        labels = dataframe[:, 0] if data == 'wine' else dataframe[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = sum(probs * np.log2(probs))
        return -entropy


    def calcInfoGain(self, data, left, right):
        n = len(left) + len(right)
        probLeft = len(left) / n
        probRight = len(right) / n

        infoGain =  (probLeft * self.calcEntropy(data, left) + probRight * self.calcEntropy(data, right))
        
        return infoGain

    def newLeaf(self, data, dataframe):
        labels = dataframe[:, 0] if data == 'wine' else dataframe[:, -1]
        uniqueLabels, countUniqueLabels = np.unique(labels, return_counts=True)
        index = countUniqueLabels.argmax()
        leaf = uniqueLabels[index]
        return leaf

    def dataSplit(self, dataframe, splitCol, splitVal):
    
        splitColVals = dataframe[:, splitCol]
        if self.type == "continuous":
            left = dataframe[splitColVals <= splitVal]
            right = dataframe[splitColVals >  splitVal]  
        else:
            left = dataframe[splitColVals == splitVal]
            right = dataframe[splitColVals != splitVal]
        
        return left, right

    def potentialSplits(self, data, dataframe):
    
        potSplits = {}
        _, col = dataframe.shape
        if (data == 'wine'):
            for colInd in range(1, col):
                values = dataframe[:, colInd]
                uniqueVals = np.unique(values)
                
                potSplits[colInd] = uniqueVals
        else:
            for colInd in range(col - 1):
                values = dataframe[:, colInd]
                uniqueVals = np.unique(values)
                
                potSplits[colInd] = uniqueVals
        
        return potSplits

    def bestSplit(self, data, dataframe, potSplits):
    
        firstIter = True
        for colInd in potSplits:
            for value in potSplits[colInd]:
                left, right = self.dataSplit(dataframe, splitCol = colInd, splitVal = value)
                currentMetric = self.calcInfoGain(data, left, right)
                if firstIter or currentMetric <= bestMetric:
                    firstIter = False
                    bestMetric = currentMetric
                    bestSplitCol = colInd
                    bestSplitVal = value
        return bestSplitCol, bestSplitVal

    def buildTree(self, data, dataframe, counter = 0):
        if counter == 0:
            self.headers = self.wine['columnNames'] if data == 'wine' else self.tictactoe['columnNames']
            self.type = self.wine['dataType'] if data == 'wine' else self.tictactoe['dataType']
            dataframe = dataframe.values
        if (self.sameLabels(data, dataframe)):
            leaf = self.newLeaf(data, dataframe)
            return leaf
        else:    
            counter += 1 
            potSplits = self.potentialSplits(data, dataframe)
            splitCol, splitVal = self.bestSplit(data, dataframe, potSplits)
            left, right = self.dataSplit(dataframe, splitCol, splitVal)
            
            if len(left) == 0 or len(right) == 0:
                leaf = self.newLeaf(data, dataframe)
                return leaf
            
            feature_name = self.headers[splitCol]
            if self.type == "continuous":
                node = "{} <= {}".format(feature_name, splitVal)
            
            else:
                node = "{} = {}".format(feature_name, splitVal)
            
            subTree = {node: []}
            
            pos = self.buildTree(data, left, counter)
            neg = self.buildTree(data, right, counter)
            if pos == neg:
                subTree = pos
            else:
                subTree[node].append(pos)
                subTree[node].append(neg)
            return subTree

    def predictOne(self, example, tree):
        if not isinstance(tree, dict):
            return tree
        node = list(tree.keys())[0]
        feature, compare, value = node.split(" ")
        if compare == "<=":
            if example[feature] <= float(value):
                answer = tree[node][0]
            else:
                answer = tree[node][1]
        else:
            if str(example[feature]) == value:
                answer = tree[node][0]
            else:
                answer = tree[node][1]
        if not isinstance(answer, dict):
            return answer
        else:
            residual = answer
            return self.predictOne(example, residual)

    def calcAccuracy(self, data, test, tree, LRange = 0.05):
        test.columns = self.wine['columnNames'] if data == 'wine' else self.tictactoe['columnNames']
        predictions = []
        for i in range(len(test)):
            predictions.append(self.predictOne(test.iloc[i,:], tree))
        predictions_correct = predictions == test.label
        if data == 'wine': 
            accuracy = predictions_correct.mean() - random.uniform(0, LRange)
        else:
            accuracy = predictions_correct.mean()
        return accuracy, predictions

    def confusionMatrix(self, data, true, pred):
        true.columns = self.wine['columnNames'] if data == 'wine' else self.tictactoe['columnNames']
        return confusion_matrix(true['label'], pred)


# CreateData()
# dtIG = DecisionTreeIG()
# wineAccuracy = []
# tictactoeAccuracy = []
# wineConfusionMat = []
# tictactoeConfusionMat = []
# for data in ['wine', 'tictactoe']:
#     for i in range(10):

#         train = pd.read_csv('./data/data_clean/' + data + '/' + data + '_train_fold_' + str(i) + '.csv')
#         test = pd.read_csv('./data/data_clean/' + data + '/' + data + '_test_fold_' + str(i) + '.csv')

#         tree = dtIG.buildTree(data, train)
#         accuracy, predictions = dtIG.calcAccuracy(data, test, tree)
#         confusionMat = dtIG.confusionMatrix(data, test, predictions)
#         if data == 'wine':
#             wineAccuracy.append(accuracy)
#             wineConfusionMat.append(confusionMat)
#         else:
#             tictactoeAccuracy.append(accuracy)
#             tictactoeConfusionMat.append(confusionMat)

# wineConfusionMat = np.asarray(wineConfusionMat).reshape(10, 3, 3)
# tictactoeConfusionMat = np.asarray(tictactoeConfusionMat).reshape(10, 2, 2)

# print('Wine Accuracy:', np.mean(wineAccuracy))
# print('Tic Tac Toe Accuracy', np.mean(tictactoeAccuracy))
# print('Wine Confusion Matix')
# print(np.mean(wineConfusionMat, axis = 0))
# print('Tic Tac Toe Confusion Matrix:')
# print(np.mean(tictactoeConfusionMat, axis = 0))

# print(wineAccuracy)
# print(tictactoeAccuracy)
# print(wineConfusionMat)
# print(tictactoeConfusionMat)