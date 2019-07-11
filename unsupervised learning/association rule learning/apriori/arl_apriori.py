

### ### PROBLEM ### ###

# create added value for customers in store X by optimising the sales of the products in the store
    








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
   


### ANALYSE ###

    # INITIALISE
    
        # create list of transactions
        # convert dataset into a list of lists

    # TRAIN APRIORI ALGORITHM
    
        # take transactions as input and produce rules as output

    
    
### REPORT ###

    # visualise results by converting the rules to a list
        
   

    







### ### IMPLEMENTATION ### ###

#%reset -f


### FIND ###

 
    # import libraries

import sys
import pandas as pd
import numpy as np
import sklearn
import scipy

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt





    # load dataset
"""
url = 
name_list = []
dataset = pd.read_csv(url, names = name_list)
"""
    
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)








### EXPLORE ###

    # summarize

  
print(dataset.shape)

print(dataset.head(10))

print(dataset.describe())

print(dataset.groupby(0).size())







### PREPARE ###

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
         
  






### ANALYSE ###

    # INITIALISE
    
        # create list of transactions

transactions = []
# remember, the apriori function will accept only a list of lists

    

        # convert dataset into a list of lists
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


    
    # TRAIN APRIORI ALGORITHM

from apyori import apriori

        # take transactions as input and produce rules as output
        
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length  = 2)

# always remember to agree the apriori arguments above with your client

# min_length refers to minimum number of items that we desire for each transaction in each rule
# min_support is specified so that the items that appear in the rules will have a support higher than the min_support
    # what support do we desire of the items in the transactions so that the rules are relevant?
    # for this we need to start with products purchased frequently eg 3 to 4 times a day
    # if we find strong rules (relationships) between items purchased 3 to 4 times a day, then by placing them together, customers are likely to buy more of them
    # so we will consider these kinds of products first, look at the resulting rules and, if we aren't happy with the rules, change the min_support and try again
    # just as we will experiment with min_support to derive realistic/satiisfactory rules, we may also experiment with min_confidence and try the rules within different periods of time and check the revenue each time until we are satisfied
    # continue until we find the strongest rules that optimise the sales
    # min_support is therefore specified as (3 transactions * 7 days) / (7500 transactions in one week) = 0.0028
# min_confidence: if your min_confidence is too high, you will obtain rules between items which don't have a strong relationship but which end up in the same basket because they are both frequently purchased items
# min_confidence: if your min_confidence is too high, you might obtain no rules at all, because there will be no relationships that occur at such a high frequency




### REPORT ###

    # visualise results by converting the rules to a list
        
results = list(rules)



results_list = []
for i in range(0, len(results)):
    results_list.append('RULE: ' + str(results[i][0]) + '\nSUPPORT: ' + str(results[i][1]) + '\nOTHERS: ' + str(results[i][2]))



# frozenset({'chicken', 'light cream'}) means that if people buy chicken then they are likely to buy light cream
    # support=0.005865884548726837 means that (the combination of?) both items appears in 0.6% of transactions
    # confidence=0.3728813559322034 means that they are 37.3% likely to buy light cream
    # lift=4.700811850163794 means that there is a 4.7 magnitude of improvement (from the support to the confidence?)











### ### NOTES ### ###



### beer-diaper connection

# company did some analytics around what people were purchasing

# analysed thousands of check outs

# they discovered that during certain times of the day, people who buy diapers also buy beer

# possibly on an excel sheet they had a tab for each product in their store

# on that tab, in one column, they probably listed each time that product was purchased

# in subsequent columns they probably then listed what other products were purchased in that transaction

# for each product they would then have seen what other product was purchased most frequently in the period under consideration








### a priori

# theoretical; theoretically; not from observation

# theoretical deductions

# made prior to observation

# "a priori (theoretical) assumptions about human nature"

# "hunger may be a factor but it cannot be assumed a priori"








### rule

# look at your data

# look at events that occur together or events that are caused by the same entity

# create rules that indicate how frequently one event is connected to another

# you'll find that some rules will be weak (low frequency connections) whilst some will be strong (high frequency connections)







### reasoning behind a priori algorithm

# algorithm consists of support, confidence and lift

# support for y
    # prediction that y will happen
    
    # support for y = (events containing y) / (total number of events)


# confidence for x and y 
    # prediction that when x happens, y will happen (rule that when x happens, y will happen)
    
    # confidence for x and y = (events containing both x and y) / (number of events containing only x)


# lift 
    # the lift is a change in your prediction that y will happen
    
    # it is either an improvement in your prediction or a worsening of your prediction 
    
    # lift for events containing both x and y = (confidence for x and y) / (support for y)







### steps to implement a priori algorithm
    
    # set up a minimum support and confidence
    
    # look in the transactions, take all the subsets that have higher support than minimum
    
    # look in the subsets, take all the rules that have confidence higher than the minimum
    
    # sort the rules by decreasing lift








### ### SAMPLE PROBLEM ### ###

# create added value for customers in store X by optimising the sales of the products in the store
    
# do this by using the algorithm to decide where to place the products
    
# buyers who probably intend only to buy A might buy A and B if the algorithm helps you to place A and B together

# see: recommender systems, collaborative filtering
