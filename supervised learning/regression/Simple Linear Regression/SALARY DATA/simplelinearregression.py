# DATA PRE PROCESSING
# REGRESSION
# CLASSIFICATION
# CLUSTERING
# ASSOCIATION RULE LEARNING
# REINFORCEMENT LEARNING
# NLP
# DEEP LEARNING
# DR
# MODEL SELECTION AND BOOSTING


# 1.0 DATA PRE PROCESSING

# PLAN

# import libraries
# set folder as working directory
# import dataset
# extract matrix of features and vector of outcomes
# OPTIONAL - estimate missing data
# OPTIONAL - encode categorical data
# OPTIONAL - create dummy variables and avoid the dummy variable trap
# create train/test split
# OPTIONAL - feature scaling

# outliers?
# junk?





# import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# set folder as working directory

    # save this file in desired folder
    # in file explorer window, navigate to directory
    # in the 'run' navigation bar above, click on run (the green 'play' button), or
    # press F5


    
    
# import dataset
    
dataset = pd.read_csv('Salary_Data.csv')

    # change format from scientific notation to float
    
        # in variable explorer window, double click on dataset
        # in variable explorer pop up window that appears, click on 'Format'
        # in the 'Format' pop up window that appears, change %.3g to %.0f
        # g represents scientific notation
        # 3 and 0 represent the number of decimal places





# extract matrix of features and vector of outcomes
        
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values





"""
# OPTIONAL - estimate missing data

    # import the Imputer class to estimate missing data
from sklearn.preprocessing import Imputer

    # replace missing data by the mean of the columns # to see the full array, run np.set_printoptions(threshold = np.nan)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

    # fit the imputer to the columns that contain missing data
imputer = imputer.fit(X[:, 1:3])

    # transform the missing data of the matrix using the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""




"""
# OPTIONAL - encode categorical data  - because ml models are based on math equations and therefore the equations require numbers to function properly

    # import LabelEncoder class to encode the categorical variables for you
from sklearn.preprocessing import LabelEncoder

    # create LabelEncoder object to encode the features
labelencoder_X = LabelEncoder()

    # fit transform the categorical values in the matrix
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

    # create LabelEncoder object to encode the outcomes - onehotencoding not necessary (ml algorithm will know the the vector of outcomes contains categories only and not scalars)
labelencoder_y = LabelEncoder()

    # fit transform the categorical values in the vector 
y = labelencoder_y.fit_transform(y)
"""





"""
# OPTIONAL - create dummy variables - to prevent algorithm from attributing order to the categorical codes

    # import OneHotEncoder class to create the dummy variables
from sklearn.preprocessing import OneHotEncoder

    # create OneHotEncoder object
onehotencoder  = OneHotEncoder(categorical_features = [0])
    
    # fit transform the categorical valeus in the matrix
X = onehotencoder.fit_transform(X).toarray()
"""





# split your data into training set and test set

    # import train_test_split class to help you
from sklearn.model_selection import train_test_split

    # create object of the train_test_split class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # model is simple therefore we can use a 1/3 ratio for test:training sets
        



        
"""
# feature scaling 
        
    # import standard scaler class to help you
from sklearn.preprocessing import StandardScaler

    # create StandardScaler object for your matrix of features X
sc_X = StandardScaler()

    # fit transform all the values in the entire training dataset and the entire test dataset 
X_train = sc_X.fit_transform(X_train) # fit to train first so that train and test are scaled on the same basis
X_test = sc_X.transform(X_test) # no need to fit sc_X to the test set, already fitted to the training set 

"""





# 2.0 REGRESSION
# 2.1 SIMPLE LINEAR REGRESSION 



# PLAN

# import LinearRegression library
# create LinearRegression object
# fit the LinearRegression object to the training set
# predict test set results
# visualize training set
# visualize test
# ?calculate accuracy of predicted test data ((actual - predicted)/predcted) * 100




# import LinearRegression library

from sklearn.linear_model import LinearRegression




# create LinearRegression object

regressor = LinearRegression()




# fit the LinearRegression object to the training set (the object is our machine, here we train our machine)

regressor.fit(X_train, y_train)




# predict test set results # create vector of predictions, y_predicted

y_predicted = regressor.predict(X_test) # A GRAPH OF X_test VS y_predicted ACTUALLY GIVES YOU THE BEST FIT LINE OF THE LEAST SQUARES METHODS





# visualize training set

    # plot the training points
plt.scatter(X_train, y_train, color = 'red')

    # visualize the regression line (plot using the training or test sets, makes no difference) this line is the predictor
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

    # add properties to the graph
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')

    #show
plt.show()

    # discuss
        # the points are close to the line, therefore the salary has a linear dependency on the years of experience





# visualize test set
    
    # plot the test points
plt.scatter(X_test, y_test, color='green')

    # visualize the regression line (plot using the training or test sets, makes no difference) this line is the predictor
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

    # add properties
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')

    # show
plt.show()

    # discuss
        # the line is seen to fit well to the test data





# some random data
        
    # my random data
d = np.array([[34], [400], [98], [22], [67]])

    # predict based on my random data
y_predicted = regressor.predict(d)

    # visualize my random data
plt.plot(d, y_predicted, color='yellow')
plt.show()




# some more random data!


test = regressor.predict([[5]])


print(test)

