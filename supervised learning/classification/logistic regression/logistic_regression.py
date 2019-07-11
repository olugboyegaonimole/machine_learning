


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
    
    # extract
    
        # extract features, extract target variable

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

    # SPLIT
        
        # create train_test_split

    # CROSS VALIDATION
        
        # calculate (mean, std) and visualize (box plot) cross_val_score for a selection of relevant algorithms with a view to choosing the best algorithm to solve this problem
        # using boxplot visualize cross_val_score array for each of the algorithms selected
        # choose model with highest cross_val_score
	            
    # MODEL BOOSTING, PARAMATER TUNING (define a function for this purpose)

        # MODEL BOOSTING
            # create gridsearchCV object
            # fit to training data
            # return object.best_estimator_

        # PARAMETER TUNING
            # learned parameters
            # hyperparameters
   
    
    
    # CLASSIFICATION
    
        # IF BOOSTING FUNCTION WAS DEFINED ABOVE
            # call function 
            # predict
        
        
        # IF BOOSTING FUNCTION WAS NOT DEFINED
            # create classification object
            # fit 
            # predict

    
    # REGULARIZATION

        # (RIDGE REGRESSION)?
        # LASSO
        # ELASTIC NET


### REPORT ###
	
    # test performance, test goodness of fit
    
        # accuracy score
        # confusion matrix
        # classification report
        # (adjusted R^2)?

    # visualise results

        # matplotlib
    
    # perform random test
    
        


    





### ### IMPLEMENTATION ### ###



### FIND ###

 
    # import libraries
    

import sys
import pandas as pd
import numpy as np
import sklearn
import scipy

from pandas.plotting import scatter_matrix
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
    
#url = 
#name_list = []
#dataset = pd.read_csv(url, names = name_list)


dataset = pd.read_csv('Social_Network_Ads.csv')



    



### EXPLORE ###

    # summarize

  
print(dataset.shape)

print(dataset.head(10))

print(dataset.describe())

print(dataset.groupby('Age').size())



    # visualize

dataset.plot(kind='box', subplots='True', layout=(2,2), sharex = False, sharey = False)
plt.show()


dataset.hist()
plt.show()


scatter_matrix(dataset)
plt.show()





### PREPARE ###

    # extract
    
        # extract features, extract target variable
     
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

""" OR
array = dataset.values

X = array[, ]
y = array[, ]
"""

    
    # clean
    
        # missing
        # invalid
        # infinity
        # duplicate
        # outlier
    
    # convert
    
        # add features
        # reduce dimensions
        # aggregate features eg create aggregates to remove noise and variability
        # disaggregate features eg from daily totals, segment into categories (oranges, apples, bananas) and create categorical totals (total yearly oranges, total yearly apples, total yearly bananas)
        
        
        
        # encode features - label encode, one hot encode, dummy variable trap
"""
from sklearn.preprocessing import LabelEncoder    
        
le = LabelEncoder()

X[:, 1] = le.fit_transform(X[:, 1])



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = [1])

X = ohe.fit_transform(X).toarray()
        

"""
        
        # scale features

from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()

ss_y = StandardScaler()

X = ss_X.fit_transform(X)

#y = y.reshape(-1,1)

#y = ss_y.fit_transform(y)







### ANALYSE ###

  
    # SPLIT

        # create train_test_split

from sklearn.model_selection import train_test_split

validation_size = 0.25
seed = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)





    # CROSS VALIDATION
        
        # calculate (mean, std) and visualize (box plot) cross_val_score for a selection of relevant algorithms with a view to choosing the best algorithm to solve this problem
        # using boxplot visualize cross_val_score array for each of the algorithms selected
        # choose model with highest cross_val_score
	            
    
    
    # MODEL BOOSTING, PARAMATER TUNING (define a function for this purpose)

        # MODEL BOOSTING
            # create gridsearchCV object
            # fit to training data
            # return object.best_estimator_


        # PARAMETER TUNING
            # learned parameters
            # hyperparameters

    
    
    # CLASSIFICATION
    
        # IF BOOSTING FUNCTION WAS DEFINED ABOVE
            # call function 
            # predict
        
        
        # IF BOOSTING FUNCTION WAS NOT DEFINED
            # create classification object
classifier = LogisticRegression(random_state = 0)


            # fit 
#y_train = y_train.reshape(-1,1)

classifier.fit(X_train, y_train)


            # predict
y_predicted = classifier.predict(X_test)

    
    
    # REGULARIZATION

        # (RIDGE REGRESSION)?
        # LASSO
        # ELASTIC NET





### REPORT ###
	
    # test performance, test goodness of fit
	
    	# accuracy score
        # confusion matrix
cm = confusion_matrix(y_test, y_predicted)

print(cm)

        # classification report
        # (adjusted R^2)?

    # visualise results

        # matplotlib
        
from matplotlib.colors import ListedColormap


X_set, y_set = X_train, y_train # X_set and y_set are not built-in; they are arbitrary variables created to allow for easy mass duplication of the X and y variable
    

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) # 0.01 represents the resolution of the pixels  # each of the pixels is an observation (not one of the observations is our dataset, but a dummy observation as it were)   # we therefore apply our classifier model to predict if each of those pixel points has value 0 or 1    # if the prediction is 0, the pixel is colored red and if the prediction is 1 the pixel is colored green

# X_set[:, 0].min() - 1 (X_set[:, 0].min() is the minimum value of the feature in the first column of the feature matrix, the -1 gives an extra space around the plot for aesthetics)
# X_set[:, 0].min() + 1 (X_set[:, 0].max() is the maximum value of the feature in the first column of the feature matrix, the +1 gives an extra space around the plot for aesthetics)
# X_set[:, 1].min() - 1 (X_set[:, 1].min() is the minimum value of the feature in the second column of the feature matrix, the -1 gives an extra space around the plot for aesthetics)
# X_set[:, 1].min() + 1 (X_set[:, 1].max() is the maximum value of the feature in the second column of the feature matrix, the +1 gives an extra space around the plot for aesthetics)


plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# plt.contour plots the contour between the two regions
# the predict function is used to predict if the selected pixel should be colored red or green

plt.xlim(X1.min(), X1.max()) # plot the limits of the first feature


plt.ylim(X2.min(), X2.max()) # plot the limits of the second feature


# the loop below plots the data points
for i,j in enumerate(np.unique(y_set)):
    
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()





    # perform random test


import random

for i in range(100):
    
    a = random.randint(2, 62)
    
    b = random.randint(5000, 97000)

    my_data = np.array([[a, b]])
    
    my_data = ss_X.transform(my_data)
    
    my_prediction = classifier.predict(my_data)
    
    print('age is {}, salary is {} and prediction is {}'.format(a, b, my_prediction))

