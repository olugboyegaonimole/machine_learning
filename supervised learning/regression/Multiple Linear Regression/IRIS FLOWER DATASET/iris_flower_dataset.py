# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:15:18 2018

@author: onimo
"""

### CLIENT REQUEST ###

### FIND ###

### EXPLORE ###

### PREPARE ###

### ANALYSE ###

### REPORT ###

### REPORT TO CLIENT ###






### CLIENT REQUEST ###

# USING THE DATASET AVAILABLE AT 'https://en.wikipedia.org/wiki/Iris_flower_data_set' PLEASE DERIVE AN ALGORITHM WHICH WILL BE USED TO CORRECTLY PREDICT THE SPECIES OF AN IRIS FLOWER IF GIVEN THE SEPAL LENGTH, SEPAL WIDTH, PETAL LENGTH AND PETAL WIDTH

# PLEASE CALCULATE AND DISPLAY THE FOLLOWING FOR THE ALGORITHM SO FOUND: 
    # CLASSIFICATION REPORT, 
    # ACCURACY SCORE, 
    # CONFUSION MATRIX

# THANK YOU
    
    
    
    

# install libraries

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




### FIND ###

    # database
    # nosql
    # csv
    # spreadsheet
    # web service
    # web socket
    # api


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
name_list = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pd.read_csv(url, names = name_list)




### EXPLORE ###

# summarize

    # shape
    # head
    # describe
    # groupby.size


# visualize
    
    # univariate - box, histogram
    # multivariate - scatter


print(dataset.shape)

print(dataset.head(10))

print(dataset.describe())

print(dataset.groupby('species').size())


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
    
        # select - remove, combine, create
        # reduce dimension
        # transform eg create aggregate to remove noise and variability
        # manipulate eg from daily totals, segment into categories (oranges, apples, bananas) and create categorical totals (total yearly oranges, total yearly apples, total yearly bananas)
        # encode - label encode, one hot encode, dummy variable trap
        # scale





### ANALYSE ###
        
    # extract features, extract dependent variable
    # create validation set
    # calculate cross_val_score (mean, std)
    # visualize cross_val_score (box plot)
    # choose model with highest cross_val_score
    # create object, fit, predict
    
    


    # extract features, extract dependent variable

X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values

""" OR
array = dataset.values

X = array[, ]
y = array[, ]
"""




    # create validation set
    
from sklearn.model_selection import train_test_split

validation_size = 0.2
seed = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)





    # calculate cross_val_score (mean, std)
    
estimators = []


estimators.append(('lr', LogisticRegression()))
estimators.append(('decision tree', DecisionTreeClassifier()))
estimators.append(('k neighbours', KNeighborsClassifier()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
estimators.append(('GaussianNB', GaussianNB()))
estimators.append(('SVC', SVC()))



cv_score_arrays = []
names = []
mean_cv_scores = []

seed = 7
scoring = 'accuracy'

maximum = ''
counter = 0

for name, estimator in estimators:
    
    cross_validator = model_selection.KFold(n_splits = 10, random_state = 7)
    
    cv_score_array = model_selection.cross_val_score(estimator, X_train, y_train, cv = cross_validator, scoring = 'accuracy')
    
    
    names.append(name)
    
    cv_score_arrays.append(cv_score_array)
    
    mean_cv_scores.append(cv_score_array.mean())
    
    
    msg = "%s: %f (%f)" % (name, cv_score_array.mean(), cv_score_array.std())
    
    if cv_score_array.mean() > counter:
        
        maximum = name
        
        counter = cv_score_array.mean()
    
    print(msg)

    
    
    # visualize cross_val_score (box plot)

figure1 = plt.figure() # create a new figure
figure1.suptitle('cross validation scores') # add a centred title


axis = figure1.add_subplot(111) # create an axis, add it as a subplot, specify the positioni of the subplot


plt.boxplot(cv_score_arrays) # plot a box plot in the figure


axis.set_xticklabels(names) # set xticklabels for the axis


plt.show()
    
    
    # choose model with highest cross_val_score
    
print("choice of algorithm is {} with a mean cv_score of {}".format(maximum, max(mean_cv_scores)))
    
    



    # create object, fit, predict
    
lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)

y_predicted = lda.predict(X_test)






### REPORT ###

# accuracy score
# confusion matrix
# classification report


print('the accuracy score is \n {}'.format(accuracy_score(y_test, y_predicted)))

print('the confusion matrix is \n {}'.format(confusion_matrix(y_test, y_predicted)))

print('the classification report is \n {}'.format(classification_report(y_test, y_predicted)))




### REPORT TO CLIENT ###

# THE ACCURACY SCORE IS 100%

# THE CONFUSION MATRIX AND CLASSIFICATION REPORT SHOW THAT EVERY PREDICTION RESULT IS PRECISE


