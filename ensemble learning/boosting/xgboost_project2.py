
### PLAN
 
# import   
    # import libraries
    # import dataset

# clean

# explore

# prepare
    # extract
    # improve
        # encode categorical data
        # avoid dummy variable trap
        
# analyse
    # choose best model
    # tune parameters
    # predict 
    # check accuracy




### IMPLEMENTATION

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display




# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values




# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
        

    # encode for countries
le_X_1 = LabelEncoder()
X[:, 1] = le_X_1.fit_transform(X[:, 1])


    # encode for genders
le_X_2 = LabelEncoder()
X[:, 2] = le_X_2.fit_transform(X[:, 2])


    # one hot encode for countries
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()


    # no need to one hot encode for genders; 
    # this is because genders has two categories; 
    # after one hot encoding we would remove one of the categories to avoid the dummy variable trap; 
    # that would leave us with one category;
    # therefore we will leave it unchanged as one category



# avoid dummy variable trap for countries
X = X[:, 1:]
        





# split into training and test
from sklearn.model_selection import train_test_split

validation_size = 0.2
seed = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)





# tune hyperparameters

from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier



def fit_model(X, y):
    
    params = {"max_depth":[1,2,3,4,5,6,7,8,9,10]} 
    
    #scoring_fnc = make_scorer(0.9) # wrong, can't call float
    
    classifier = XGBClassifier()
    
    cv_sets = ShuffleSplit(n_splits=10, test_size = 0.20, random_state = 0)
    
    grid = GridSearchCV(estimator = classifier, param_grid=params, scoring='accuracy', cv=cv_sets)
    
    #X.astype(int) # if needed
    
    grid = grid.fit(X, y)
    
    return grid.best_estimator_





# fit model to the training set (not required if using parameter tuning)   
model = fit_model(X_train, y_train)




# predict
y_predicted = model.predict(X_test)

	
    	

# check accuracy using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)




# check accuracy using k-fold cross validation

    # this will give you the average accuracy from 10 accuracies (ie from 10 training and 10 test folds)

from sklearn import model_selection

cv_score_array = model_selection.cross_val_score(estimator=model, X=X_train, y=y_train, cv = 10)
    
cv_score_array.mean() # the average accuracy from 10 accuracies 

cv_score_array.std() # the standard deviation from the average accuracy above: multiply by 100 to obtain the value in percent
    








### NOTES


# qualities of xgboost

    # high performance on huge datasets
    # high speed
    # you can keep the interpretation of your dataset, your model and your results


# xgboost is a gradient boosting model with decision tress, therefore feature scaling is unnecessary



# The xgboost library provides a system for use in a range of computing environments, not least:
    # Parallelization of tree construction using all of your CPU cores during training
    # Distributed Computing for training very large models using a cluster of machines (see https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
    
    
    
# by way of comparison, view implementation of adaboost below
"""  
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1) #here I have used decision tree as a base estimator, you can use any ML learner as base estimator if it accepts sample weight 
clf.fit(x_train,y_train)

source: https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/
"""




# remember to install xgboost on your pc by using one of the following in git bash

    # OPTION 1
    # 'pip3 install xgboost'
    
    # OR
    
    # OPTION 2
    # 'anaconda search -t conda xgboost' (here you will find the name and version of the xgboost package appropriate for your pc) followed by 
    # 'conda install -c prefix suffix=version' (where name of package can be seen in the result of the command above as 'prefix/suffix')




