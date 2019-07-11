### ### PROBLEM ### ###

# segment the customers of a mall into groups on the basis of two known metrics:

    # their annual income
    # their spending score at the mall
    
    
    
    
#%reset -f and then CTRL + l in the console to reset variable explorer and clear console
    
    

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

        # extract features

    
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


    # ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS
    
        
    # APPLY ALGORITHM TO DATASET USING OPTIMAL NUMBER ACQUIRED FROM PREVIOUS STEP
  

### REPORT ###
	
    # visualise clusters

        # matplotlib


    







### ### IMPLEMENTATION ### ###



### FIND ###

 
    # import libraries


import sys
import pandas as pd #to import and manage datasets
import numpy as np #to use mathematics
import sklearn
import scipy

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection







    # load dataset
    
"""url = 
name_list = []
dataset = pd.read_csv(url, names = name_list)
"""
    
dataset = pd.read_csv('Mall_Customers.csv')





### EXPLORE ###

    # summarize

  
print(dataset.shape)

print(dataset.head(10))

print(dataset.describe())

print(dataset.groupby('Spending Score (1-100)').size())



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
     
X = dataset.iloc[:, [3,4]].values
#y = dataset.iloc[, ].values


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
"""
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

"""





### ANALYSE ###

  
    # ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS

from sklearn.cluster import KMeans

wcss = [] # within clusters sum of squares


for i in range(1, 11):
    
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init=10, random_state = 0)
# init - the method for initialisation
# n_init - the number of times the algorithm will be run with different initial centroids
# random_state - fixes all the random factors of the kmeans process
        
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
    

plt.plot(range(1,11), wcss)
plt.title('elbow')
plt.xlabel('clusters')
plt.ylabel('wcss')
plt.show()

# the graph reveals that the elbow occurs at 5 clusters
    
    
    



    # APPLY ALGORITHM TO DATASET USING OPTIMAL NUMBER ACQUIRED FROM PREVIOUS STEP

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init=10, random_state = 0)

y_clusters = kmeans.fit_predict(X)
 
    
    



### REPORT ###
	
    # visualise clusters

plt.scatter(X[y_clusters == 0, 0], X[y_clusters == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_clusters == 1, 0], X[y_clusters == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_clusters == 2, 0], X[y_clusters == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_clusters == 3, 0], X[y_clusters == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_clusters == 4, 0], X[y_clusters == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('client clusters')

plt.xlabel('annual income $000')

plt.ylabel('spending score 1-100')

plt.legend()

plt.show()








### ### NOTES ### ###
    
    
    
### clustering steps ###

# choose the number, k, of clusters you want

# randomly select k points each to serve as a centroid for each cluster

# assign each point to the closest centroid, this forms k clusters

# loop

    # if necessary, re-calculate the centroid of each cluster based on the coordinates of the points in each cluster
    
    # re-assign each datapoint to the new closest centroid

# end
    
    
    
    
    
    
    
### random initialisation trap ###
    
# the positions of your centroids, selected randomly at the beginning of the clustering operation, can sometimes cause you to have less accurate clusters than you would have had if another set of centroids had been selected

# to avoid this trap use the k-means++ algorithm







### WCSS ###
    
# we need a metric to help evaluate how a number of clusters performs compared to a any other number of clusters

# the metric should be quantifiable
    
# within clusters sum of squares
    
# for each cluster, square the distance from the centroid to each point and sum the squares. Add up the sums

# you can have as many points as they are clusters. At that stage WCSS will = 0






### the WCSS elbow method ###
    
# plot a graph of WCSS vs number of clusters

# the graph looks like a bent arm

# the optimal number of clusters is the point at the elbow
    
    
    
    
    

