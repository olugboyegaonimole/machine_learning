# AIM to estimate the salary of an employee when provided with the grade level of the employee within the organisation

# OBJECTIVE to show that this is possible  using a machine learning algorithm


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp


	# load dataset

	    # database
	    # nosql
	    # csv
	    # spreadsheet
	    # web service
	    # web socket
	    # api

dataset = pd.read_csv('Position_Salaries.csv')




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
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


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
from sklearn.tree import  DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)


	    # fit (ONLY IF MODEL BOOSTING NOT USED ABOVE)
regressor.fit(X, y)


	    # call model boosting function (IF MODEL BOOSTING WAS USED ABOVE)


	    # predict
y_predicted = regressor.predict(6.5)





### REPORT ###
	
	# test performance
	
		# accuracy score
		# confusion matrix
		# classification report

	# visualise results

		# matplotlib

			# visualize regression results
pp.scatter(X, y, color='red')
pp.plot(X, regressor.predict(X), color='blue')
pp.show()


			# visualise decision tree results (using a higher resolution for a smoother curve, ie proper segmentation of intervals)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
pp.scatter(X, y, color='red')
pp.plot(X_grid, regressor.predict(X_grid), color='blue')
pp.show()



"""

pp.scatter(X, y, color='red')
pp.plot(X, regressor.predict(X), color='green')
pp.show()
""" 

    
			# my real world data
y_predicted2 = regressor.predict(6.8)



    


### NOTES

# Decision tree - non-linear and non-continuous

    # The decision tree algorithm considers entropy and information gained and splits the independent variables into several intervals.
    # In each interval the algorithm uses the average of the dependent variable. 
    # This average is a constant, hence the graph is represented by a straight line throughout each interval segment.