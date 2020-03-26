import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class CreateNoisyLabel:
    def __init__(self):
        self.mu = 0
        self.sigma = 1
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
        
        self.wineDataframe = self.readData('wine')
        self.tictactoeDataframe = self.readData('tictactoe')
        self.makeDirs()
        self.writeData()
        
    def makeDirs(self):

        if not os.path.exists('./data/label_dirty/wine/con/five/'):os.makedirs('./data/label_dirty/wine/con/five/')
        if not os.path.exists('./data/label_dirty/wine/con/ten/'):os.makedirs('./data/label_dirty/wine/con/ten/')
        if not os.path.exists('./data/label_dirty/wine/con/fifteen/'):os.makedirs('./data/label_dirty/wine/con/fifteen/')

        if not os.path.exists('./data/label_dirty/wine/mis/five/'):os.makedirs('./data/label_dirty/wine/mis/five/')
        if not os.path.exists('./data/label_dirty/wine/mis/ten/'):os.makedirs('./data/label_dirty/wine/mis/ten/')
        if not os.path.exists('./data/label_dirty/wine/mis/fifteen/'):os.makedirs('./data/label_dirty/wine/mis/fifteen/')

        if not os.path.exists('./data/label_dirty/tictactoe/con/five/'):os.makedirs('./data/label_dirty/tictactoe/con/five/')
        if not os.path.exists('./data/label_dirty/tictactoe/con/ten/'):os.makedirs('./data/label_dirty/tictactoe/con/ten/')
        if not os.path.exists('./data/label_dirty/tictactoe/con/fifteen/'):os.makedirs('./data/label_dirty/tictactoe/con/fifteen/')

        if not os.path.exists('./data/label_dirty/tictactoe/mis/five/'):os.makedirs('./data/label_dirty/tictactoe/mis/five/')
        if not os.path.exists('./data/label_dirty/tictactoe/mis/ten/'):os.makedirs('./data/label_dirty/tictactoe/mis/ten/')
        if not os.path.exists('./data/label_dirty/tictactoe/mis/fifteen/'):os.makedirs('./data/label_dirty/tictactoe/mis/fifteen/')

    def readData(self, data):
        if data == 'wine':
            dataframe = pd.read_csv(self.wine['data'], names = self.wine['columnNames'])
        elif data == 'tictactoe':
            dataframe = pd.read_csv(self.tictactoe['data'], names = self.tictactoe['columnNames'])
        else:
            print('Wrong data file')
            return 
        return dataframe.sample(frac = 1)

    def tictactoeHelper(self, current):
        rand = np.random.randint(0, 1)
        if current == 'x':
            if rand == 0: return 'o'
            else: return 'b'
        elif current == 'o':
            if rand == 0: return 'x'
            else: return 'b' 
        elif current == 'b':
            if rand == 0: return 'x'
            else: return 'o'

    def wineHelper(self, current):
        rand = np.random.randint(0, 1)
        if current == 1:
            if rand == 0: return 2
            else: return 3
        elif current == 2:
            if rand == 0: return 1
            else: return 3
        elif current == 3:
            if rand == 0: return 1
            else: return 2

    def addConExample(self, data, dataframe, L):
        limit = int((len(dataframe.index) + 1) * L)
        if data == 'wine':
            for i in range(limit):
                row = dataframe.iloc[i]
                row.label = self.wineHelper(row.label)
                dataframe = dataframe.append(row)
        else:
            for i in range(limit):
                row = dataframe.iloc[i]
                if row.label == 'positive': row.label = 'negative'
                elif row.label == 'negative': row.label = 'positive'
                dataframe = dataframe.append(row)
        return dataframe
    
    def misExample(self, data, dataframe, L):
        limit = int((len(dataframe.index) + 1) * L)
        if data == 'wine':
            for i in range(limit):
                current = dataframe.iloc[i]['label']
                new = self.wineHelper(current)
                dataframe.iloc[i]['label'] = new
        else:
            for i in range(limit):
                current = dataframe.iloc[i]['label']
                if current == 'positive': new = 'negative'
                else: new = 'positive'
                dataframe.iloc[i]['label'] = new
        return dataframe
                


    def writeData(self, n = 10):
        LName = ['five', 'ten', 'fifteen']
        LRange = [0.05, 0.10, 0.15]
        wineDataframe = self.wineDataframe
        tictactoeDataframe = self.tictactoeDataframe
        cv = KFold(n_splits = n, shuffle = True)
        
        data = 'wine'
        #con example
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(wineDataframe):
                train, test = wineDataframe.loc[train_index], wineDataframe.loc[test_index]
                train = self.addConExample(data, train, LRange[j])
                train.to_csv('./data/label_dirty/wine/con/'+ LName[j] + '/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/label_dirty/wine/con/' + LName[j] + '/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
                
        #misslabeld example
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(wineDataframe):
                train, test = wineDataframe.loc[train_index], wineDataframe.loc[test_index]
                train = self.misExample(data, train, LRange[j])
                train.to_csv('./data/label_dirty/wine/mis/'+ LName[j] + '/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/label_dirty/wine/mis/' + LName[j] + '/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        
        data = 'tictactoe'
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(tictactoeDataframe):
                train, test = tictactoeDataframe.loc[train_index], tictactoeDataframe.loc[test_index]
                train = self.addConExample(data, train, LRange[j])
                train.to_csv('./data/label_dirty/tictactoe/con/'+ LName[j] + '/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/label_dirty/tictactoe/con/' + LName[j] + '/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
                
        #misslabeld example
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(tictactoeDataframe):
                train, test = tictactoeDataframe.loc[train_index], tictactoeDataframe.loc[test_index]
                train = self.misExample(data, train, LRange[j])
                train.to_csv('./data/label_dirty/tictactoe/mis/'+ LName[j] + '/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/label_dirty/tictactoe/mis/' + LName[j] + '/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        print('Done creating data for Q3B')
pd.options.mode.chained_assignment = None
# CreateNoisyLabel()