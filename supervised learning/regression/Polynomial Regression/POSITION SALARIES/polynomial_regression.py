# DATA PRE PROCESSING
# FEATURE ENGINEERING
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
    
dataset = pd.read_csv('Position_Salaries.csv')

    # change format from scientific notation to float
    
        # in variable explorer window, double click on dataset (type is listed as DataFrame)
        # in variable explorer if type is listed as object it means the matrix contains many types and cannot select a single type, hence cannot display the data
        # in variable explorer pop up window that appears, click on 'Format'
        # in the 'Format' pop up window that appears, change %.3g to %.0f
        # g represents scientific notation
        # 3 and 0 represent the number of decimal places





# extract matrix of features and vector of outcomes
        
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values




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






# splitting not necessary here
    # observations are too few 
    # we need high accuracy; therefore we need maximum info so that model can perfectly see all the correlations
"""
# split your data into training set and test set

    # import train_test_split class to help you
from sklearn.cross_validation import train_test_split

    # create object of the train_test_split class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
""" 




   
# feature scaling not necessary here
    # we only need to add polynomial terms to multiple linear regression equation
    # therefore we will use same library as for simple and multiple linear regression
    # the library (statsmodel.formula.api? or sklearn.linear_model?) does feature scaling for itself
"""
# feature scaling 
        
    # import StandardScaler class to help you
from sklearn.preprocessing import StandardScaler

    # create StandardScaler object for your matrix of features X
sc_X = StandardScaler()

    # fit transform all the values in the entire training dataset and the entire test dataset 
X_train = sc_X.fit_transform(X_train) # fit to train first so that train and test are scaled on the same basis
X_test = sc_X.transform(X_test) # no need to fit sc_X to the test set, already fitted to the training set 
"""     
  






# 2.0 REGRESSION
# 2.1 SIMPLE LINEAR REGRESSION
# 2.2 MULTIPLE LINEAR REGRESSION
# 2.3 POLYNOMIAL LINEAR REGRESSION






# 2.3 POLYNOMIAL LINEAR REGRESSION


# PLAN

# fit linear regression model to the entire dataset (for future comparison purposes only)
# using PolynomialFeatures object, fit_transform the features matrix into a polynomial dataset
# fit regression model to the polynomial dataset 
# visualize actual points and straight line prediction
# visualize actual points and polynomial line prediction





# fit linear regression model to the entire dataset (for future comparison purposes only)

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)





# using Polynomial Features object, fit_transform features matrix into a polynomial dataset

    # import class PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

    # create an object of the PolynomialFeatures class, specify degree
polynomial_features = PolynomialFeatures(degree = 7)

    # using the object, fit_transform the features matrix into a polynomial dataset
X_poly = polynomial_features.fit_transform(X)





# fit regression model to the polynomial dataset 

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)




# visualize actual points and straight line predictions 

plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.show()




# visualize actual points and polynomial line predictions 

plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor_2.predict(X_poly), color='green')
plt.show()