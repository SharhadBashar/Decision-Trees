import pandas as pd
from pandas import DataFrame 
import pprint
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class DecisionTreeGR():
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.wine = {
            'data': 'wine.data',
            'dataType': 'continuous',
            'labelInd': 0,
            'length': 178,
            'columnNames': ['label', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color', 'hue', 'od280', 'proline']
        }
        self.tictactoe = {
            'data': 'tic-tac-toe.data',
            'dataType': 'categorical',
            'labelInd': -1,
            'length': 958,
            'columnNames': ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br', 'label']
        }

    def calcEntropy(self, dataframe):
        label = dataframe.keys()[-1]
        entropy = 0
        values = dataframe[label].unique()
        for value in values:
            temp = dataframe[label].value_counts()[value] / len(dataframe[label])
            entropy += temp * np.log2(temp)
        return -entropy
 
    def calcGainRatio(self, dataframe, att):
        eps = self.eps
        label = dataframe.keys()[-1]   
        targetVars = dataframe[label].unique()  
        variables = dataframe[att].unique()   
        entropy = 0
        for variable in variables:
            tempEntropy = 0
            for targetVar in targetVars:
                num = len(dataframe[att][dataframe[att] == variable][dataframe[label] == targetVar])
                den = len(dataframe[att][dataframe[att] == variable])
                temp = num / (den + eps)
                tempEntropy += -temp * np.log2(temp + eps)
            temp2 = den / len(dataframe)
            entropy += -temp2 * tempEntropy
        return abs(entropy)


    def bestSplit(self, dataframe):
        GR = []
        for key in dataframe.keys()[:-1]:
            GR.append(self.calcEntropy(dataframe) - self.calcGainRatio(dataframe, key))
        return dataframe.keys()[:-1][np.argmax(GR)]



    def buildTree(self, dataframe, tree = None): 
        label = dataframe.keys()[-1]
        node = self.bestSplit(dataframe)

        attValue = np.unique(dataframe[node])

        if tree is None:                    
            tree = {}
            tree[node] = {} 

        for value in attValue:
            
            subtable = dataframe[dataframe[node] == value].reset_index(drop = True)
            clValue,counts = np.unique(subtable['label'], return_counts = True)                        
            
            if len(counts) == 1:
                tree[node][value] = clValue[0]                                                    
            else:        
                tree[node][value] = self.buildTree(subtable) 
        return tree

    def predictOne(self, example, tree, default = 1):
        for key in list(example.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][example[key]] 
                except:
                    return default
                result = tree[key][example[key]]
                if isinstance(result,dict):
                    return self.predictOne(example,result)

                else:
                    return result

    def calcAccuracy(self, test, tree):
        predictions = []
        queries = test.iloc[:,:-1].to_dict(orient = "records")
        for i in range(len(test)):
            predictions.append(self.predictOne(queries[i], tree))
        predictions_correct = predictions == test.label
        accuracy = predictions_correct.mean()
        return accuracy, predictions

    def confusionMatrix(self, data, true, pred):
        return confusion_matrix(true['label'], pred)

# namesTicTacToe =['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br','label']
# namesWine = ['label', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color', 'hue', 'od280', 'proline']
# dtGR = DecisionTreeGR()

# wineAccuracy = []
# tictactoeAccuracy = []
# wineConfusionMat = []
# tictactoeConfusionMat = []
# for data in ['wine', 'tictactoe']:
#     for i in range(10):

#         train = pd.read_csv('./data/data_clean/' + data + '/' + data + '_train_fold_' + str(i) + '.csv', names = namesWine if data == 'wine' else namesTicTacToe)
#         test = pd.read_csv('./data/data_clean/' + data + '/' + data + '_test_fold_' + str(i) + '.csv', names = namesWine if data == 'wine' else namesTicTacToe)

#         tree = dtGR.buildTree(train)
#         accuracy, predictions = dtGR.calcAccuracy(test, tree)
#         confusionMat = dtGR.confusionMatrix(data, test, predictions)
#         if data == 'wine':
#             wineAccuracy.append(accuracy)
#             wineConfusionMat.append(confusionMat)
#         else:
#             tictactoeAccuracy.append(accuracy)
#             tictactoeConfusionMat.append(confusionMat)
#
# print('Wine Accuracy:', np.mean(wineAccuracy))
# print('Tic Tac Toe Accuracy', np.mean(tictactoeAccuracy))
# print('Wine Confusion Matix')
# print(np.mean(wineConfusionMat, axis = 0))
# print('Tic Tac Toe Confusion Matrix:')
# print(np.mean(tictactoeConfusionMat, axis = 0))
#
# print(wineAccuracy)
# print(tictactoeAccuracy)
# print(wineConfusionMat)
# print(tictactoeConfusionMat)