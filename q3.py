# coding: utf-8
import numpy as np
import csv
import pandas as pd
import random
from pprint import pprint
import matplotlib.pyplot as plt
#function
from create_noisy_data import CreateNoisyData
from create_noisy_label import CreateNoisyLabel 
from decision_tree_IG import DecisionTreeIG

####################################################################################   3A    ####################################################################################
wine3A = {'five': {'CvsC': [], 'CvsD': [], 'DvsC': [], 'DvsD': []}, 
       'ten': {'CvsC': [], 'CvsD': [], 'DvsC': [], 'DvsD': []}, 
       'fifteen': {'CvsC': [], 'CvsD': [], 'DvsC': [], 'DvsD': []}
}

tictactoe3A = {'five': {'CvsC': [], 'CvsD': [], 'DvsC': [], 'DvsD': []}, 
            'ten': {'CvsC': [], 'CvsD': [], 'DvsC': [], 'DvsD': []}, 
            'fifteen': {'CvsC': [], 'CvsD': [], 'DvsC': [], 'DvsD': []}
}

def run3A():
    #Prep data
    CreateNoisyData()
    dtIG = DecisionTreeIG()
    testType = ['CvsC', 'CvsD', 'DvsC', 'DvsD']
    LName = ['five', 'ten', 'fifteen']
    LRange = [0.05, 0.1, 0.15]
    for data in ['wine', 'tictactoe']:
        for i in range(3): #LName
            for j in range(4): #testType
                for k in range(10):
                    train = pd.read_csv('./data/data_dirty/' + data + '/' + LName[i] + '/' + testType[j] + '/train_fold_' + str(k) + '.csv')
                    test = pd.read_csv('./data/data_dirty/' + data + '/' + LName[i] + '/' + testType[j] + '/test_fold_' + str(k) + '.csv')
                    tree = dtIG.buildTree(data, train)
                    accuracy, _ = dtIG.calcAccuracy(data, test, tree, LRange[i] + 0.1)
                    if data == 'wine':
                        wine3A[LName[i]][testType[j]].append(accuracy)
                    else:
                        tictactoe3A[LName[i]][testType[j]].append(accuracy)
    return wine3A, tictactoe3A

def mean(data):
    return np.mean(data) * 100
def variance(data):
    return np.var(data)

def plot3A(wine, tictactoe):
    print('Wine CvsC Accuracy (mean) with L = 5% :', mean(wine['five']['CvsC']), '%')
    print('Wine CvsC Accuracy (variance) with L = 5% :', variance(wine['five']['CvsC']))
    print('Wine CvsC Accuracy (mean) with L = 10% :', mean(wine['ten']['CvsC']), '%')
    print('Wine CvsC Accuracy (variance) with L = 10% :', variance(wine['ten']['CvsC']))
    print('Wine CvsC Accuracy (mean) with L = 15% :', mean(wine['fifteen']['CvsC']), '%')
    print('Wine CvsC Accuracy (variance) with L = 15% :', variance(wine['fifteen']['CvsC']))

    print('Wine CvsD Accuracy (mean) with L = 5% :', mean(wine['five']['CvsD']), '%')
    print('Wine CvsD Accuracy (variance) with L = 5% :', variance(wine['five']['CvsD']))
    print('Wine CvsD Accuracy (mean) with L = 10% :', mean(wine['ten']['CvsD']), '%')
    print('Wine CvsD Accuracy (variance) with L = 10% :', variance(wine['ten']['CvsD']))
    print('Wine CvsD Accuracy (mean) with L = 15% :', mean(wine['fifteen']['CvsD']), '%')
    print('Wine CvsD Accuracy (variance) with L = 15% :', variance(wine['fifteen']['CvsD']))

    print('Wine DvsC Accuracy (mean) with L = 5% :', mean(wine['five']['DvsC']), '%')
    print('Wine DvsC Accuracy (variance) with L = 5% :', variance(wine['five']['DvsC']))
    print('Wine DvsC Accuracy (mean) with L = 10% :', mean(wine['ten']['DvsC']), '%')
    print('Wine DvsC Accuracy (variance) with L = 10% :', variance(wine['ten']['DvsC']))
    print('Wine DvsC Accuracy (mean) with L = 15% :', mean(wine['fifteen']['DvsC']), '%')
    print('Wine DvsC Accuracy (variance) with L = 15% :', variance(wine['fifteen']['DvsC']))

    print('Wine DvsD Accuracy (mean) with L = 5% :', mean(wine['five']['DvsD']), '%')
    print('Wine DvsD Accuracy (variance) with L = 5% :', variance(wine['five']['DvsD']))
    print('Wine DvsD Accuracy (mean) with L = 10% :', mean(wine['ten']['DvsD']), '%')
    print('Wine DvsD Accuracy (variance) with L = 10% :', variance(wine['ten']['DvsD']))
    print('Wine DvsD Accuracy (mean) with L = 15% :', mean(wine['fifteen']['DvsD']), '%')
    print('Wine DvsD Accuracy (variance) with L = 15% :', variance(wine['fifteen']['DvsD']))

    wineCvsC = np.array([[5, mean(wine['five']['CvsC'])], [10, mean(wine['ten']['CvsC'])], [15, mean(wine['fifteen']['CvsC'])]]).T
    wineCvsD = np.array([[5, mean(wine['five']['CvsD'])], [10, mean(wine['ten']['CvsD'])], [15, mean(wine['fifteen']['CvsD'])]]).T
    wineDvsC = np.array([[5, mean(wine['five']['DvsC'])], [10, mean(wine['ten']['DvsC'])], [15, mean(wine['fifteen']['DvsC'])]]).T
    wineDvsD = np.array([[5, mean(wine['five']['DvsD'])], [10, mean(wine['ten']['DvsD'])], [15, mean(wine['fifteen']['DvsD'])]]).T

    plt.plot(wineCvsC[0], wineCvsC[1], label = 'CvsC')
    plt.plot(wineCvsD[0], wineCvsD[1], label = 'CvsD')
    plt.plot(wineDvsC[0], wineDvsC[1], label = 'DvsC')
    plt.plot(wineDvsD[0], wineDvsD[1], label = 'DvsD')
    plt.legend()
    plt.title('Wine Data set')
    plt.xlabel('L %')
    plt.ylabel('Accuracy %')
    plt.show()


    print('Tic Tac Toe CvsC Accuracy (mean) with L = 5% :', mean(tictactoe['five']['CvsC']), '%')
    print('Tic Tac Toe CvsC Accuracy (variance) with L = 5% :', variance(tictactoe['five']['CvsC']))
    print('Tic Tac Toe CvsC Accuracy (mean) with L = 10% :', mean(tictactoe['ten']['CvsC']), '%')
    print('Tic Tac Toe CvsC Accuracy (variance) with L = 10% :', variance(tictactoe['ten']['CvsC']))
    print('Tic Tac Toe CvsC Accuracy (mean) with L = 15% :', mean(tictactoe['fifteen']['CvsC']), '%')
    print('Tic Tac Toe CvsC Accuracy (variance) with L = 15% :', variance(tictactoe['fifteen']['CvsC']))

    print('Tic Tac Toe CvsD Accuracy (mean) with L = 5% :', mean(tictactoe['five']['CvsD']), '%')
    print('Tic Tac Toe CvsD Accuracy (variance) with L = 5% :', variance(tictactoe['five']['CvsD']))
    print('Tic Tac Toe CvsD Accuracy (mean) with L = 10% :', mean(tictactoe['ten']['CvsD']), '%')
    print('Tic Tac Toe CvsD Accuracy (variance) with L = 10% :', variance(tictactoe['ten']['CvsD']))
    print('Tic Tac Toe CvsD Accuracy (mean) with L = 15% :', mean(tictactoe['fifteen']['CvsD']), '%')
    print('Tic Tac Toe CvsD Accuracy (variance) with L = 15% :', variance(tictactoe['fifteen']['CvsD']))

    print('Tic Tac Toe DvsC Accuracy (mean) with L = 5% :', mean(tictactoe['five']['DvsC']), '%')
    print('Tic Tac Toe DvsC Accuracy (variance) with L = 5% :', variance(tictactoe['five']['DvsC']))
    print('Tic Tac Toe DvsC Accuracy (mean) with L = 10% :', mean(tictactoe['ten']['DvsC']), '%')
    print('Tic Tac Toe DvsC Accuracy (variance) with L = 10% :', variance(tictactoe['ten']['DvsC']))
    print('Tic Tac Toe DvsC Accuracy (mean) with L = 15% :', mean(tictactoe['fifteen']['DvsC']), '%')
    print('Tic Tac Toe DvsC Accuracy (variance) with L = 15% :', variance(tictactoe['fifteen']['DvsC']))

    print('Tic Tac Toe DvsD Accuracy (mean) with L = 5% :', mean(tictactoe['five']['DvsD']), '%')
    print('Tic Tac Toe DvsD Accuracy (variance) with L = 5% :', variance(tictactoe['five']['DvsD']))
    print('Tic Tac Toe DvsD Accuracy (mean) with L = 10% :', mean(tictactoe['ten']['DvsD']), '%')
    print('Tic Tac Toe DvsD Accuracy (variance) with L = 10% :', variance(tictactoe['ten']['DvsD']))
    print('Tic Tac Toe DvsD Accuracy (mean) with L = 15% :', mean(tictactoe['fifteen']['DvsD']), '%')
    print('Tic Tac Toe DvsD Accuracy (variance) with L = 15% :', variance(tictactoe['fifteen']['DvsD']))

    tictactoeCvsC = np.array([[5, mean(tictactoe['five']['CvsC'])], [10, mean(tictactoe['ten']['CvsC'])], [15, mean(tictactoe['fifteen']['CvsC'])]]).T
    tictactoeCvsD = np.array([[5, mean(tictactoe['five']['CvsD'])], [10, mean(tictactoe['ten']['CvsD'])], [15, mean(tictactoe['fifteen']['CvsD'])]]).T
    tictactoeDvsC = np.array([[5, mean(tictactoe['five']['DvsC'])], [10, mean(tictactoe['ten']['DvsC'])], [15, mean(tictactoe['fifteen']['DvsC'])]]).T
    tictactoeDvsD = np.array([[5, mean(tictactoe['five']['DvsD'])], [10, mean(tictactoe['ten']['DvsD'])], [15, mean(tictactoe['fifteen']['DvsD'])]]).T

    plt.plot(tictactoeCvsC[0], tictactoeCvsC[1], label = 'CvsC')
    plt.plot(tictactoeCvsD[0], tictactoeCvsD[1], label = 'CvsD')
    plt.plot(tictactoeDvsC[0], tictactoeDvsC[1], label = 'DvsC')
    plt.plot(tictactoeDvsD[0], tictactoeDvsD[1], label = 'DvsD')
    plt.legend()
    plt.title('Tic Tac Toe Data set')
    plt.xlabel('L %')
    plt.ylabel('Accuracy %')
    plt.show()


####################################################################################   3A    ####################################################################################

####################################################################################   3B    ####################################################################################
wine3B = {'con': {'five': [], 'ten': [], 'fifteen': []}, 
        'mis': {'five': [], 'ten': [], 'fifteen': []},
}

tictactoe3B = {'con': {'five': [], 'ten': [], 'fifteen': []}, 
             'mis': {'five': [], 'ten': [], 'fifteen': []}
}

def run3B():
    #Prep data
    CreateNoisyLabel()
    dtIG = DecisionTreeIG()
    testType = ['con', 'mis']
    LName = ['five', 'ten', 'fifteen']
    LRange = [0.05, 0.1, 0.15]
    for data in ['wine', 'tictactoe']:
        for i in range(3): #LName
            for j in range(2): #testType
                for k in range(10):
                    train = pd.read_csv('./data/label_dirty/' + data + '/' + testType[j] + '/' + LName[i] + '/train_fold_' + str(k) + '.csv')
                    test = pd.read_csv('./data/label_dirty/' + data + '/' + testType[j] + '/' + LName[i] + '/test_fold_' + str(k) + '.csv')
                    tree = dtIG.buildTree(data, train)
                    accuracy, _ = dtIG.calcAccuracy(data, test, tree, LRange[i] + 0.1)
                    if data == 'wine':
                        wine3B[testType[j]][LName[i]].append(accuracy)
                    else:
                        tictactoe3B[testType[j]][LName[i]].append(accuracy)
    return wine3B, tictactoe3B

def mean(data):
    return np.mean(data) * 100

def plot3B(wine, tictactoe):
    wineCon = np.array([[5, mean(wine['con']['five'])], [10, mean(wine['con']['ten'])], [15, mean(wine['con']['fifteen'])]]).T
    wineMis = np.array([[5, mean(wine['mis']['five'])], [10, mean(wine['mis']['ten'])], [15, mean(wine['mis']['fifteen'])]]).T
    
    plt.plot(wineCon[0], wineCon[1], label = 'Contradictory')
    plt.plot(wineMis[0], wineMis[1], label = 'Misclassification')
    plt.legend()
    plt.title('Wine Data set')
    plt.xlabel('L %')
    plt.ylabel('Accuracy %')
    plt.show()

    tictactoeCon = np.array([[5, mean(tictactoe['con']['five'])], [10, mean(tictactoe['con']['ten'])], [15, mean(tictactoe['con']['fifteen'])]]).T
    tictactoeMis = np.array([[5, mean(tictactoe['mis']['five'])], [10, mean(tictactoe['mis']['ten'])], [15, mean(tictactoe['mis']['fifteen'])]]).T
    plt.plot(tictactoeCon[0], tictactoeCon[1], label = 'Contradictory')
    plt.plot(tictactoeMis[0], tictactoeMis[1], label = 'Misclassification')
    plt.legend()
    plt.title('Tic Tac Toe Data set')
    plt.xlabel('L %')
    plt.ylabel('Accuracy %')
    plt.show()

def run():
    wine3A, tictactoe3A = run3A()
    plot3A(wine3A, tictactoe3A)

    wine3B, tictactoe3B = run3B()
    plot3B(wine3B, tictactoe3B)

run()