import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class CreateNoisyData:
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
        
    def addNoise(self, data, dataframe, L):
        limit = int((len(dataframe.index) + 1) * L)
        if data == 'wine':
            for i in range(1, len(self.wine['columnNames'])):
                noise = np.random.normal(self.mu, self.sigma, 1)
                header = self.wine['columnNames'][i]
                dataframe.loc[:limit, header] += noise
            return dataframe
        else:
            for i in range(limit):
                for j in range(len(self.tictactoe['columnNames']) - 1):
                    current = dataframe.iloc[i][self.tictactoe['columnNames'][j]]
                    new = self.tictactoeHelper(current)
                    dataframe.iloc[i][self.tictactoe['columnNames'][j]] = new
            return dataframe
        
    def makeDirs(self):
        
        if not os.path.exists('./data/data_dirty/wine/five/CvsC/'): 
            os.makedirs('./data/data_dirty/wine/five/CvsC/')
        if not os.path.exists('./data/data_dirty/wine/five/CvsD/'):
            os.makedirs('./data/data_dirty/wine/five/CvsD/')
        if not os.path.exists('./data/data_dirty/wine/five/DvsC/'):
            os.makedirs('./data/data_dirty/wine/five/DvsC/')
        if not os.path.exists('./data/data_dirty/wine/five/DvsD/'):
            os.makedirs('./data/data_dirty/wine/five/DvsD/')
        
        if not os.path.exists('./data/data_dirty/wine/ten/CvsC/'):os.makedirs('./data/data_dirty/wine/ten/CvsC/')
        if not os.path.exists('./data/data_dirty/wine/ten/CvsD/'):os.makedirs('./data/data_dirty/wine/ten/CvsD/')
        if not os.path.exists('./data/data_dirty/wine/ten/DvsC/'):os.makedirs('./data/data_dirty/wine/ten/DvsC/')
        if not os.path.exists('./data/data_dirty/wine/ten/DvsD/'):os.makedirs('./data/data_dirty/wine/ten/DvsD/')

        if not os.path.exists('./data/data_dirty/wine/fifteen/CvsC/'):os.makedirs('./data/data_dirty/wine/fifteen/CvsC/')
        if not os.path.exists('./data/data_dirty/wine/fifteen/CvsD/'):os.makedirs('./data/data_dirty/wine/fifteen/CvsD/')
        if not os.path.exists('./data/data_dirty/wine/fifteen/DvsC/'):os.makedirs('./data/data_dirty/wine/fifteen/DvsC/')
        if not os.path.exists('./data/data_dirty/wine/fifteen/DvsD/'):os.makedirs('./data/data_dirty/wine/fifteen/DvsD/')

        if not os.path.exists('./data/data_dirty/tictactoe/five/CvsC/'):os.makedirs('./data/data_dirty/tictactoe/five/CvsC/')
        if not os.path.exists('./data/data_dirty/tictactoe/five/CvsD/'):os.makedirs('./data/data_dirty/tictactoe/five/CvsD/')
        if not os.path.exists('./data/data_dirty/tictactoe/five/DvsC/'):os.makedirs('./data/data_dirty/tictactoe/five/DvsC/')
        if not os.path.exists('./data/data_dirty/tictactoe/five/DvsD/'):os.makedirs('./data/data_dirty/tictactoe/five/DvsD/')

        if not os.path.exists('./data/data_dirty/tictactoe/ten/CvsC/'):os.makedirs('./data/data_dirty/tictactoe/ten/CvsC/')
        if not os.path.exists('./data/data_dirty/tictactoe/ten/CvsD/'):os.makedirs('./data/data_dirty/tictactoe/ten/CvsD/')
        if not os.path.exists('./data/data_dirty/tictactoe/ten/DvsC/'):os.makedirs('./data/data_dirty/tictactoe/ten/DvsC/')
        if not os.path.exists('./data/data_dirty/tictactoe/ten/DvsD/'):os.makedirs('./data/data_dirty/tictactoe/ten/DvsD/')

        if not os.path.exists('./data/data_dirty/tictactoe/fifteen/CvsC/'):os.makedirs('./data/data_dirty/tictactoe/fifteen/CvsC/')
        if not os.path.exists('./data/data_dirty/tictactoe/fifteen/CvsD/'):os.makedirs('./data/data_dirty/tictactoe/fifteen/CvsD/')
        if not os.path.exists('./data/data_dirty/tictactoe/fifteen/DvsC/'):os.makedirs('./data/data_dirty/tictactoe/fifteen/DvsC/')
        if not os.path.exists('./data/data_dirty/tictactoe/fifteen/DvsD/'):os.makedirs('./data/data_dirty/tictactoe/fifteen/DvsD/')
        
    
    def writeData(self, n = 10):
        LName = ['five', 'ten', 'fifteen']
        LRange = [0.05, 0.10, 0.15]
        wineDataframe = self.wineDataframe
        tictactoeDataframe = self.tictactoeDataframe
        cv = KFold(n_splits = n, shuffle = True)
        #wine
        #CvsC
        data = 'wine'
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(wineDataframe):
                train, test = wineDataframe.loc[train_index], wineDataframe.loc[test_index]
                train.to_csv('./data/data_dirty/wine/'+ LName[j] + '/CvsC/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/wine/' + LName[j] + '/CvsC/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        #DvsC
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(wineDataframe):
                train, test = wineDataframe.loc[train_index], wineDataframe.loc[test_index]
                train = self.addNoise(data, train, LRange[j])
                train.to_csv('./data/data_dirty/wine/'+ LName[j] + '/DvsC/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/wine/' + LName[j] + '/DvsC/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
                
        #CvsD
        
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(wineDataframe):
                train, test = wineDataframe.loc[train_index], wineDataframe.loc[test_index]
                test = self.addNoise(data, test, LRange[j])
                train.to_csv('./data/data_dirty/wine/'+ LName[j] + '/CvsD/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/wine/' + LName[j] + '/CvsD/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        #DvsD
        
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(wineDataframe):
                train, test = wineDataframe.loc[train_index], wineDataframe.loc[test_index]
                train = self.addNoise(data, train, LRange[j])
                test = self.addNoise(data, test, LRange[j])
                train.to_csv('./data/data_dirty/wine/'+ LName[j] + '/DvsD/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/wine/' + LName[j] + '/DvsD/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
                
        #tictactoe
        #CvsC
        data = 'tictactoe'
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(tictactoeDataframe):
                train, test = tictactoeDataframe.loc[train_index], tictactoeDataframe.loc[test_index]
                train.to_csv('./data/data_dirty/tictactoe/'+ LName[j] + '/CvsC/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/tictactoe/' + LName[j] + '/CvsC/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        #DvsC
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(tictactoeDataframe):
                train, test = tictactoeDataframe.loc[train_index], tictactoeDataframe.loc[test_index]
                train = self.addNoise(data, train, LRange[j])
                train.to_csv('./data/data_dirty/tictactoe/'+ LName[j] + '/DvsC/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/tictactoe/' + LName[j] + '/DvsC/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
                
        #CvsD
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(tictactoeDataframe):
                train, test = tictactoeDataframe.loc[train_index], tictactoeDataframe.loc[test_index]
                test = self.addNoise(data, test, LRange[j])
                train.to_csv('./data/data_dirty/tictactoe/'+ LName[j] + '/CvsD/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/tictactoe/' + LName[j] + '/CvsD/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        #DvsD
        for j in range(3):
            i = 0
            for train_index, test_index in cv.split(tictactoeDataframe):
                train, test = tictactoeDataframe.loc[train_index], tictactoeDataframe.loc[test_index]
                train = self.addNoise(data, train, LRange[j])
                test = self.addNoise(data, test, LRange[j])
                train.to_csv('./data/data_dirty/tictactoe/'+ LName[j] + '/DvsD/train_fold_' + str(i) + '.csv', index = False, header = False)
                test.to_csv('./data/data_dirty/tictactoe/' + LName[j] + '/DvsD/test_fold_' + str(i) + '.csv', index = False, header = False)
                i += 1
        print('Done creating data for Q3A')
# CreateNoisyData()