# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:15:18 2018

@author: onimo
"""


### ### PLAN ### ###


### FIND ###

	# import libraries

	# load dataset

	    # database
	    # nosql
	    # csv
	    # spreadsheet
	    # web service
	    # web socket
	    # api


### EXPLORE ###

	# summarize

	    # shape
	    # head
	    # describe
	    # groupby.size

	# visualize
	    
	    # univariate - box, histogram
	    # multivariate - scatter


### PREPARE ###

	# clean
    
        # missing
        # invalid
        # infinity
        # duplicate
        # outlier
    
    # transform
    
        # add features
        # reduce dimensions
        # aggregate features eg create aggregates to remove noise and variability
        # disaggregate features eg from daily totals, segment into categories (oranges, apples, bananas) and create categorical totals (total yearly oranges, total yearly apples, total yearly bananas)
        # encode features - label encode, one hot encode, dummy variable trap
        # scale features


### ANALYSE ###

    # CROSS VALIDATION
   
	   	# extract features, extract target variable
	    # create train_test_split
	    # calculate (mean, std) and visualize (box plot) cross_val_score for a selection of relevant algorithms with a view to choosing the best algorithm to solve this problem
	    # using boxplot visualize cross_val_score array for each of the algorithms selected
	    # choose model with highest cross_val_score
	            
    # MODEL BOOSTING (define a function for this purpose)

    	# create gridsearchCV object
    	# fit to training data
    	# return object.best_estimator_

    # REGRESSION
    
	    # create object (ONLY IF MODEL BOOSTING NOT USED ABOVE)
	    # fit (ONLY IF MODEL BOOSTING NOT USED ABOVE)
	    # call model boosting function (IF MODEL BOOSTING WAS USED ABOVE)
	    # predict


### REPORT ###
	
	# test performance
	
		# accuracy score
		# confusion matrix
		# classification report

	# visualise results

		# matplotlib


    





### ### IMPLEMENTATION ### ###


### FIND ###


	# import libraries

import sys
import pandas as pd
import numpy
import sklearn
import scipy

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



    # load dataset
    
dataset = pd.read_csv('Position_Salaries.csv')





    
### EXPLORE ###


	# summarize

print(dataset.shape)

print(dataset.head(10))

print(dataset.describe())

print(dataset.groupby('Level').size())



	# visualize

dataset.plot(kind='box', subplots='True', layout=(2,2), sharex = False, sharey = False)
plt.show()


dataset.hist()
plt.show()


scatter_matrix(dataset)
plt.show()






### PREPARE ###

    # clean
    
        # missing
        # invalid
        # infinity
        # duplicate
        # outlier
    
    # transform
    
        # add features
        # reduce dimensions
        # aggregate features eg create aggregates to remove noise and variability
        # disaggregate features eg from daily totals, segment into categories (oranges, apples, bananas) and create categorical totals (total yearly oranges, total yearly apples, total yearly bananas)
        # encode features - label encode, one hot encode, dummy variable trap
        # scale features






### ANALYSE ###
        


    # CROSS VALIDATION

	   	# extract features, extract target variable

X = dataset.iloc[:, 1:2 ].values
y = dataset.iloc[:, 2 ].values


""" OR
array = dataset.values

X = array[, ]
y = array[, ]
"""

   
    	# create train_test_split   

from sklearn.model_selection import train_test_split

validation_size = 0.2
seed = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)




""" 	# calculate cross_val_score (mean, std) for a selection of algorithms
    
from sklearn import model_selection

estimators = []
estimators.append(("linear regression", LinearRegression()))
estimators.append((,))


cv_score_arrays = []
names = []
mean_cv_scores = []

seed = 7
scoring = 'accuracy'

best_estimator = "" 
counter = 0

for name, estimator in estimators:
    
    cross_validator = model_selection.KFold(n_splits = 10, random_state = 7)
    
    cv_score_array = model_selection.cross_val_score(estimator, X_train, y_train, cv = cross_validator, scoring = 'accuracy')
    
    
    names.append(name)
    
    cv_score_arrays.append(cv_score_array)
    
    mean_cv_scores.append(cv_score_array.mean())
    
    
    msg = "%s: %f (%f)" % (name, cv_score_array.mean(), cv_score_array.std())
    
    if cv_score_array.mean() > counter:
        
        best_estimator = name
        
        counter = cv_score_array.mean()
    
    print(msg)"""

    
   
     
""" 	# using boxplot visualize cross_val_score array for each of the algorithms selected
    
figure1 = plt.figure()

figure1.suptitle()

ax = figure1.add_subplot(111)

plt.boxplot(cv_score_arrays)

ax.set_xticklabels(names)

plt.show()"""




""" 	# choose model with highest cross_val_score
    
print("choice of algorithm is {} with a mean cv_score of {}".format(best_estimator, max(mean_cv_scores)))

"""
    
   

   # MODEL BOOSTING (define a function for this purpose)

    	# create gridsearchCV object
    	# fit to training data
    	# return object.best_estimator_




    # REGRESSION
    
	    # create object (ONLY IF MODEL BOOSTING NOT USED ABOVE)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state =0)

		# fit (ONLY IF MODEL BOOSTING NOT USED ABOVE)
regressor.fit(X_train, y_train)

	    # call model boosting function (IF MODEL BOOSTING WAS USED ABOVE)


    	# predict    
y_predicted = regressor.predict (X_test) 






### REPORT ###


	# test performance

print(accuracy_score(y_test, y_predicted))

print(confusion_matrix(y_test, y_predicted))

print(classification_report(y_test, y_predicted))



	# visualise results

# plt.scatter()
# plt.plot()