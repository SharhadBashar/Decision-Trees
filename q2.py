# coding: utf-8
import numpy as np
import csv
import pandas as pd
import random
from pprint import pprint
from sklearn.metrics import confusion_matrix
#functions
from create_data import CreateData
from decision_tree_IG import DecisionTreeIG
from decision_tree_GR import DecisionTreeGR

def run():
	#Prep data
	CreateData()

	#Q2 a)
	wineAccuracyIG = []
	tictactoeAccuracyIG = []
	wineConfusionMatIG = []
	tictactoeConfusionMatIG = []

	dtIG = DecisionTreeIG()

	for data in ['wine', 'tictactoe']:
	    for i in range(10):

	        train = pd.read_csv('./data/data_clean/' + data + '/' + data + '_train_fold_' + str(i) + '.csv')
	        test = pd.read_csv('./data/data_clean/' + data + '/' + data + '_test_fold_' + str(i) + '.csv')

	        tree = dtIG.buildTree(data, train)

	        accuracy, predictions = dtIG.calcAccuracy(data, test, tree)
	        confusionMat = dtIG.confusionMatrix(data, test, predictions)
	        if data == 'wine':
	            wineAccuracyIG.append(accuracy)
	            wineConfusionMatIG.append(confusionMat)
	        else:
	            tictactoeAccuracyIG.append(accuracy)
	            tictactoeConfusionMatIG.append(confusionMat)

	wineConfusionMatIG = np.asarray(wineConfusionMatIG).reshape(10, 3, 3)
	tictactoeConfusionMatIG = np.asarray(tictactoeConfusionMatIG).reshape(10, 2, 2)

	print()
	print('Wine Accuracy for Information Gain (Mean):', np.mean(wineAccuracyIG) * 100, '%')
	print()
	print('Wine Accuracy for Information Gain (Variance):', np.var(wineAccuracyIG))
	print()
	print('Tic Tac Toe Accuracy for Information Gain (Mean):', np.mean(tictactoeAccuracyIG) * 100, '%')
	print()
	print('Tic Tac Toe Accuracy for Information Gain (Variance):', np.var(tictactoeAccuracyIG))
	print()
	print('Wine Confusion Matix for Information Gain:')
	print()
	print(np.mean(wineConfusionMatIG, axis = 0))
	print()
	print('Tic Tac Toe Confusion Matrix for Information Gain:')
	print()
	print(np.mean(tictactoeConfusionMatIG, axis = 0))

	#Q2 b)
	namesTicTacToe =['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br','label']
	namesWine = ['label', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color', 'hue', 'od280', 'proline']
	wineAccuracyGR = []
	tictactoeAccuracyGR = []
	wineConfusionMatGR = []
	tictactoeConfusionMatGR = []

	dtGR = DecisionTreeGR()

	for data in ['wine', 'tictactoe']:
	    for i in range(10):

	        train = pd.read_csv('./data/data_clean/' + data + '/' + data + '_train_fold_' + str(i) + '.csv', names = namesWine if data == 'wine' else namesTicTacToe)
	        test = pd.read_csv('./data/data_clean/' + data + '/' + data + '_test_fold_' + str(i) + '.csv', names = namesWine if data == 'wine' else namesTicTacToe)

	        tree = dtGR.buildTree(train)
	        accuracy, predictions = dtGR.calcAccuracy(test, tree)
	        confusionMat = dtGR.confusionMatrix(data, test, predictions)
	        if data == 'wine':
	            wineAccuracyGR.append(accuracy)
	            wineConfusionMatGR.append(confusionMat)
	        else:
	            tictactoeAccuracyGR.append(accuracy)
	            tictactoeConfusionMatGR.append(confusionMat)

	wineConfusionMatGR = np.asarray(wineConfusionMatGR).reshape(10, 3, 3)
	tictactoeConfusionMatGR = np.asarray(tictactoeConfusionMatGR).reshape(10, 3, 3)

	print()
	print('Wine Accuracy for Gain Ratio (Mean):', np.mean(wineAccuracyGR) * 100, '%')
	print()
	print('Wine Accuracy for Gain Ratio (Variance):', np.var(wineAccuracyGR))
	print()
	print('Tic Tac Toe Accuracy for Gain Ratio (Mean):', np.mean(tictactoeAccuracyGR) * 100, '%')
	print()
	print('Tic Tac Toe Accuracy for Gain Ratio (Mean):', np.var(tictactoeAccuracyGR))
	print()
	print('Wine Confusion Matix for Gain Ratio:')
	print()
	print(np.mean(wineConfusionMatGR, axis = 0))
	print()
	print('Tic Tac Toe Confusion Matrix for Gain Ratio:')
	print()
	tictactoeConfusionMatGR = np.mean(tictactoeConfusionMatGR, axis = 0)
	tictactoeConfusionMatGR = [[tictactoeConfusionMatGR[1][1], tictactoeConfusionMatGR[1][2]],
             				   [tictactoeConfusionMatGR[2][1], tictactoeConfusionMatGR[2][2]]]
	print(tictactoeConfusionMatGR)

run()