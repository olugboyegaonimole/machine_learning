
### ### PROBLEM ### ###

    # you've been given a list of 1000 reviews written by visitors to a restaurant in which they described the quality of their experience and in which they also rated the experience 1 for 'good' or 0 for 'bad'

    # using this list of reviews, create an nlp model that can be used to accurately analyse future written reviews and judge if the customer has had a good experience or a bad one














### ### PLAN ### ###


### FIND

    # import libraries
    # load dataset

### EXPLORE

    # summarise
        
        # shape
        # head
        # groupby().size()

### PREPARE

    # clean
        
        # missing or duplicate 
        # invalid 

    # convert
        
        # add fields
        # remove fields

### ANALYSE

    # CLEAN
        
        # create corpus (empty list)
        # remove non-alphabets
        # convert to lower case
        # convert to a list
        # remove stopwords
        # stem non-stopwords
        # convert back to a string
        # append to corpus
        
    # BAG OF WORDS
        
        # convert corpus to sparse matrix (matrix of features, X) through tokenization
        # create vector of outcomes, y
        
    # CLASSIFY
    
        # create train test split
        # create classifier
        # fit
        # predict if reviews are good or bad

### REPORT
    
    # confusion matrix
    
    









### ### IMPLEMENTATION ### ###



### FIND

    # import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    # load dataset
    
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# quoting = 3 ignores all double quotes in the text








### EXPLORE

    # summarise
        
        # shape
        # head
        # groupby().size()







### PREPARE

    # clean
        
        # missing or duplicate 
        # invalid 


    # convert
        
        # add fields
        # remove fields







### ANALYSE
        
        

    # CLEAN
        
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



    
corpus = [] # this will be your cleaned dataset, it will be a list of strings. it must be a list of strings for countvectorizer to work
    



a, b = dataset.shape




for i in range(a):
    
    
        # remove non-alphabets
        
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # re.sub removes everything except alphabets - removes numbers and punctuation
    # ^a-z means not a to z, ie dont remove any alphabets
    # [^a-zA-Z]', ' ' means replace what you're removing with a space
    
    
    
    
       	# convert to lower case
    
    review = review.lower()
    
    
    
        # convert to a list, if not already a list (otherwise the next step will return a list of alphabets instead of a list of words)
        
    review = review.split()


    
        # remove stopwords
    
    review = [x for x in review if not x in set(stopwords.words('english'))]
    # use set(stopwords.words('english')) rather than the list stopwords.words('english'). This is because the algorithm will look through a set faster than it will look through a list
	
    
        
        # stem non-stopwords
            
    porterstemmer = PorterStemmer()
    
    review = [porterstemmer.stem(x) for x in review]



        # convert back to a string
        
    review = ' '.join(review)
        
    
    
        # append to list of strings
        
    corpus.append(review)






    # BAG OF WORDS - SPARSE MATRIX AND VECTOR OF OUTCOMES
    
        # CountVectorize to create sparse matrix (matrix of features, X) 

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()


        # create vector of outcomes, y

y = dataset.iloc[:, 1].values






    # CLASSIFY



        # create train test split
        
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




        # list of classification models
        
            # logistic regression
            # knn
            # svm
            # kernel svm
            # naive bayes
            # decision tree classification
            # random forest classification
            # CART
            # C5.0
            # Maximum Entropy


            # the most common models used for nlp are

                # naive bayes
                # decision tree classification, and 
                # random forest classification






        # create classifier 1 naive bayes

from sklearn.naive_bayes import GaussianNB
        

        # fit

classifier1 = GaussianNB()

classifier1.fit(X_train, y_train)
      

        # predict if reviews are good or bad

y_predicted1 = classifier1.predict(X_test)






        # create classifier 2...

        



        # create classifier 3 decision tree classification

from sklearn.tree import DecisionTreeClassifier


        # fit

classifier3 = DecisionTreeClassifier()        

classifier3.fit(X_train, y_train)

        
        # predict if reviews are good or bad

y_predicted3 = classifier3.predict(X_test)









        # create classifier 4 random forest classification

from sklearn.ensemble import RandomForestClassifier
      

        # fit

classifier4 = RandomForestClassifier(n_estimators=200)

classifier4.fit(X_train, y_train)

        
        # predict if reviews are good or bad

y_predicted4 = classifier4.predict(X_test)








### REPORT
    
    # confusion matrix

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



cm1 = confusion_matrix(y_test, y_predicted1)

print(classification_report(y_test, y_predicted1))

print(accuracy_score(y_test, y_predicted1)) # 0.73





cm3 = confusion_matrix(y_test, y_predicted3)

print(classification_report(y_test, y_predicted3))

print(accuracy_score(y_test, y_predicted3)) # 0.66





cm4 = confusion_matrix(y_test, y_predicted4)

print(classification_report(y_test, y_predicted4))

print(accuracy_score(y_test, y_predicted4)) # 0.715










### ### NOTES ### ###

# branch of computer science concerned with the interaction between computers and human languages

# a branch of computer science used to apply ml models to human languages

# text cleaning yields a bag of words

# text cleaning 

    # gets rid of punctuation 
    
    # gets rid of non-useful words

    # carries out stemming, ie extracting the root of words so that you have one word representing many words that have that word as their stem eg love for loved, loving, lover
    
    # gets rid of capital letters

# tokenization - splits text into individual words, create a table where each word has a column and each cell contains an integer telling you how many times the word appears in the selected text
    
    
    


### ### PERSONAL SUMMARY ### ###

    # in nlp we aim to classify our new data into categories

    # a category might be an alphabet (a,b,c,...) a number (9, 78, 0.45) a word written or sounded out ('london', 'jesus', 'tomato',..) a label (good, bad, spam, true, false,..) etc

    # to do this we first train an algorithm using historical data
    
    # if we are happy with the accuracy of the training, we can thereafter use the algorithm for classifying new input data
    
    # before we train our algorithm we need to prepare our historical data
    
        # create a list (corpus) to hold the observations of your independent variable    
    
        # do the following to each observation of the independent variable
        
            # remove non alphabets
            
            # convert to lower case
            
            # convert to a list
            
            # remove stopwords
            
            # stem non stopwords

            # convert back to a string
            
            # add the string to the corpus
                        
        # convert your corpus to a sparse matrix
            
        # split your sparse matrix and vector of outcomes into a training set and validation set
        

