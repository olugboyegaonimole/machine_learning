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
    
dataset = pd.read_csv('50_Startups.csv')

    # change format from scientific notation to float
        # in variable explorer window, double click on dataset
        # in variable explorer pop up window that appears, click on 'Format'
        # in the 'Format' pop up window that appears, change %.3g to %.0f
        # g represents scientific notation
        # 3 and 0 represent the number of decimal places





# extract matrix of features and vector of outcomes
        
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values






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





# OPTIONAL - encode categorical data  - because ml models are based on math equations and therefore the equations require numbers to function properly

    # import LabelEncoder class to encode the categorical variables for you
from sklearn.preprocessing import LabelEncoder

    # create LabelEncoder object to encode the features
labelencoder_X = LabelEncoder()

    # fit transform the categorical values in the matrix
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
"""
    # create LabelEncoder object to encode the outcomes - onehotencoding not necessary (ml algorithm will know the the vector of outcomes contains categories only and not scalars)
labelencoder_y = LabelEncoder()

    # fit transform the categorical values in the vector 
y = labelencoder_y.fit_transform(y)
"""






# OPTIONAL - create dummy variables - to prevent algorithm from attributing order to the categorical codes

    # import OneHotEncoder class to create the dummy variables
from sklearn.preprocessing import OneHotEncoder

    # create OneHotEncoder object
onehotencoder  = OneHotEncoder(categorical_features = [3])
    
    # fit transform the categorical valeus in the matrix
X = onehotencoder.fit_transform(X).toarray()






# avoid the dummy variable trap

X = X[:, 1:] 
    # the python library for linear regression has taken care of the dummy variable trap so we dont need to do this manually in our code
    # for some software/libraries however you need to take onevariable away manually
    # removes redundant dependencies






# split your data into training set and test set

    # import train_test_split class to help you
from sklearn.model_selection import train_test_split

    # create object of the train_test_split class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # model is simple therefore we can use a 1/3 ratio for test:training sets
        




        
"""
# feature scaling - we dont need this for multiple linear regression because the library will do that for us
        
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
# 2.2 MULTIPLE LINEAR REGRESSION


# PLAN

# fit multiple linear regressor to all features in training set (to create line of best fit)

# predict and view test results

# perform backward elimination of insignificant features (to further improve line of best fit)

    # decide value of SL

    # import statsmodels.formula.api

    # create the y intercept by appending a column of ones to our matrix X

    # define X_optimal = X[:, 0,1,2,3,...,n]
        
    # fit regressor = statsmodels.OLS().fit()
    
    # view regressor.summary() to find feature with the highest P
    
    # remove feature with highest P, refit regressor, repeat process until highest P <= SL

    # from X_optimal create a new training set and a new test set

    # fit the LinearRegression regressor to the training set

    # make a prediction using the test set






# fit multiple linear regressor to all features in training set (to create line of best fit)

    # import LinearRegression class
from sklearn.linear_model import LinearRegression

    # create object of the class
regressor = LinearRegression()
    
    # fit regressor to the training sets
regressor.fit(X_train, y_train)





# predict and view test results

y_predicted = regressor.predict(X_test)

    # verdict
        # on viewing the predictions, the predictions are seen to be reasonably close to the test figures
        # there is a multiple linear dependency between the independent variables and the dependent variable
        # we were able to fit a linear model to our dataset and it worked!
        
        
        
        
        
# perform backward elimination of insignificant features (to further improve line of best fit)

    # import statsmodels.formula.api
import statsmodels.formula.api as sm

    

    # create y intercept by appending column of ones to matrix X
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)


    # create optimal matrix and initialise with all features
X_optimal = X[:, [0,1,2,3,4,5]]
    # create OLS regressor object from statsmodels library # fit, to regressor object, matrix of all features
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    # view summary
regressor_OLS.summary()


    # remove index of predictor with highest P value if P > SL
X_optimal = X[:, [0,1,3,4,5]]
    # refit regressor 
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    # view summary
regressor_OLS.summary()


    # remove index of predictor with highest P value if P > SL
X_optimal = X[:, [0,3,4,5]]
    # refit regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    # view summary
regressor_OLS.summary()


    # remove index of predictor with highest P value if P > SL
X_optimal = X[:, [0,3,5]]
    # refit regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    # view summary
regressor_OLS.summary()


    # remove index of predictor with highest P value if P > SL
X_optimal = X[:, [0,3]]
    # refit regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
    # view summary
regressor_OLS.summary()






# from the optimal model create a new training set and a new test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_optimal, y, test_size = 0.2, random_state = 0)






# fit the LinearRegression regressor to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)






# make a prediction using the test set

y_predicted = regressor.predict(X_test)