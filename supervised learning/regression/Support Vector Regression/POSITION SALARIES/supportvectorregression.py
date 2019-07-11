# AIM to estimate the salary of an employee when provided with the grade level of the employee within the organisation

# OBJECTIVE to show that this is possible  using a machine learning algorithm



# plan

# preprocessing

    # import libraries
    # import dataset
    # extract matrix and vector
    # feature scaling
        # reshape y
        # fit_transform X and y

   
    

# regression

    # create regressor
    # fit regressor 
    # predict
        # transform X
        # predict y
        # inverse_transform y
    # visualise SVR results (use np.arange and X_grid.reshape to create X_grid for higher resolution and smoother curve)
    # discuss results






# implementation


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# import dataset
dataset = pd.read_csv('Position_Salaries.csv')



# extract matrix and vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



# feature scaling - the SVR class doesnt seem to apply feature scaling in its algorithm
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X = ss_X.fit_transform(X)

# you must reshape your y vector, because in feature scaling the fit_transform Standard Scaler method expects a 2d array and not a 1d array
y = y.reshape(-1,1)
y = ss_y.fit_transform(y) # value error: expected 2d array, got 1d array instead. reshape your data using array.reshape(-1,1) if your data has a single feature or array.reshape(1,-1) if it contains a single sample



# fit svr to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y) # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel(). # y = column_or_1d(y, warn=True)



# predict a corresponding y variable for a specified x variable

# feature scale x variable 6.5 before predicting corresponding y variable

# a = ss_X.fit_transform(6.5) # WRONG. CAN'T FIT_TRANSFORM 6.5. STANDARD SCALER SHOULD BE FITTED ONLY ONCE TO A MATRIX OR VECTOR. ALSO, EXPRESS 6.5 AS A 2D ARRAY
# a = ss_X.transform([[6.5]]) # WRONG. RESHAPE 6.5 TO 2D ARRAY USING NP.ARRAY METHOD
x = ss_X.transform(np.array([[6.5]])) # ALWAYS MAKE SURE TO USE 2D ARRAY



# predict y
y_predicted = regressor.predict(x)



# use the inverse transform to restore y to original scale (before feature scaling was performed)    
y_predicted = ss_y.inverse_transform(y_predicted)





# visualise SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1)) # array.reshape
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# discuss results

    # the line doesn't fit closely to the CEO data point
    # the CEO datapoint is considered an outlier
    # the SVR has some penalty parameters selected by default in its algorithm
    # and so because the CEO datapoint is very far from the other datapoints the algorithm considers it an outlier
    # the SVR algorithm therefore made its fit on the other datapoints