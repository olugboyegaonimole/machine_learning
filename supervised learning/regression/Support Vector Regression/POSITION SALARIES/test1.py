# plan - to test if results will be affected if feature scaling takes place before train-test split

# in this version, splitting is done before scaling

### PLAN ###


# PRE PROCESSING
    # IMPORT LIBRARIES
    # IMPORT DATASET
    # EXTRACT FEATURES, EXTRACT TARGET
    # CLEAN
    # WRANGLE
    # SPLIT

# REGRESSION
    # IMPORT CLASS
    # CREATE OBJECT
    # FIT
    # PREDICT


    
    


### IMPLEMENTATION ###


# PRE PROCESSING
    # IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp


    # IMPORT DATASET
dataset = pd.read_csv('Position_Salaries_test.csv')


    # EXTRACT FEATURES, EXTRACT OUTCOME
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


    # CLEAN
        # missing
        # duplicates
        # outliers
        # invalid

        
    
    # SPLIT
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    
        
    # WRANGLE
        # select
        # scale
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)


y_train = y_train.reshape(-1, 1)
y_train = ss_y.fit_transform(y_train)

y_test = y_test.reshape(-1, 1)
y_test = ss_y.transform(y_test)

        # transform
        # categorical
        # dr
        # manipulate
        
    
    
# REGRESSION
    # IMPORT CLASS
from sklearn.svm import SVR

    # CREATE OBJECT
regressor = SVR(kernel = 'rbf')

    # FIT
#y_train = np.ravel(y_train)
regressor.fit(X_train, y_train)

    # PREDICT
y_predicted = regressor.predict(X_test)

    # PREDICT SINGLE VARIABLE
y_pred_single = regressor.predict(ss_X.transform(np.array([[6.7]])))
y_pred_single = ss_y.inverse_transform(y_pred_single) # 9.405097323409287492e+04

   



# visualise SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_train), max(X_train), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled, choice of X_train and not X because X_train was scaled and not X
X_grid = X_grid.reshape((len(X_grid), 1)) # array.reshape
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR) - Split before Scale')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


 
"""
pp.scatter(X, y, color='blue')
pp.plot(X_train, regressor.predict(X_train), color='red')
pp.show()
"""