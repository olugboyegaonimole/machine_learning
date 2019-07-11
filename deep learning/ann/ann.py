
### ### PROBLEM ### ###

    # a bank is experiencing unusually high churn rates and wants to know why
    
    # please help
    
    # create a geo demographic segmentation model to tell the bank which customers are at highest risk of leaving
    
    # some customer data provided
    
    # (this skill is transferable to any problem with a binary outcome and lots of independent variables)
    
        
        
    
    
    
    
    
    

### ### PLAN ### ###
    
    ### FIND

    ### EXPLORE
    
    ### PREPARE
    
    ### ANALYSE
    
        # CREATE VALIDATION SET
        
        # BUILD NETWORK
        
            # INITIALISE NETWORK
            
                # create a classifier as an object of Sequential
            
            # ADD LAYERS AND TRAIN
        
                # randomly initialise your weights with small numbers close to 0
                
                # input the first observation of your dataset in the input layer, one feature per node
                
                # propagate till you get y^
                
                # measure the error (compare actual to predicted)
                
                # back propagate the error, update the weights according to howmuch they are responsible for the error
                
                    # the learning rate parameter, which you can set up for your network, will decide by how much the weights are updated
                    
                # when the whole training set has passed throu the ann you have an epoch. redo more epochs
                
        
        # FIT TO TRAINING SET
        
        # VALIDATE
        
    
    ### REPORT TO CLIENT
    
    
    
    
    
    

### ### IMPLEMENT ### ###
    



### FIND ###

 
    # import libraries


import pandas as pd
import numpy as np


from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import keras



    # load dataset
"""   
url = 
name_list = []
dataset = pd.read_csv(url, names = name_list)
"""

dataset = pd.read_csv('Churn_Modelling.csv')
    





### EXPLORE ###

    # summarize

  
print(dataset.shape)

print(dataset.head(10))

print(dataset.describe())

print(dataset.groupby('Geography').size())



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
     
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
        
le_X_1 = LabelEncoder()

X[:, 1] = le_X_1.fit_transform(X[:, 1])




le_X_2 = LabelEncoder()

X[:, 2] = le_X_2.fit_transform(X[:, 2])




ohe = OneHotEncoder(categorical_features = [1])

X = ohe.fit_transform(X).toarray()

X = X[:, 1:]
        





 







### ANALYSE ###

  
    # CREATE VALIDATION SET
    
        # create train_test_split

from sklearn.model_selection import train_test_split

validation_size = 0.2
seed = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)


       
        # scale features

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)







    # BUILD NETWORK
        
        # initialise network

from keras.models import Sequential

            # define a sequence of layers or define a graph

classifier = Sequential()
        

     



        # add layers and compile

from keras.layers import Dense


            # add input layer and first hidden layer

classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_shape=(11,))) # the add function does not add the input layer; it only adds the hidden layer; so you need to specify shape of inputss
# units - refers to the optimal no. of nodes you select for the hidden layer
    # choose the avg of the number of nodes in the hidden layer and the output layer
    # or use parameter tuning eg creating a cross validation set and using kFold cross validation
# kernel_initializer - helps initialise your weights as small numbers close to 0
# activation - helps swhich which activation function you apply to your hidden layer ('relu' for rectifier function) and which you apply to your output layer



            # add a second hidden layer

classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu')) 
# here use activation='softmax' if your dependent variable hasmore than 2 categories



            # add output layer

classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid')) 



            # compile - apply stochastic gradient descent on the whole network

classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer - refers to the stochastic gradient algorithm
# loss - refers to the loss function, C
    # if you use a sigmoid activation function for this, you obtain a logistic regression model; here the loss function will be a logrithmic loss
    # if dependent variable has more than two outcomes, loss=categorical_crossentropy
# metrics - an accuracy criterion used to improve the model's performance




    
    # FIT TO TRAINING SET

#y_train = y_train.reshape(-1,1)

classifier.fit(X_train, y_train, batch_size=10, epochs=100) 
# ACC WAS .8439 CHECK IF BETTER ACC IF FEATURE SCALING HAPPENS AFTER TRAIN TEST SPLIT
# ACC WAS .8430 WHEN FEATURE SCALING HAPPENED AFTER TRAIN TEST SPLIT




    # VALIDATE
    
        # predict

y_predicted = classifier.predict(X_test)

y_predicted = (y_predicted > 0.5)


	
        # evaluate performance
    	
            # confusion matrix
            
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

print(cm)    



        	# accuracy score

from sklearn.metrics import accuracy_score

a_s = accuracy_score(y_test, y_predicted) # ACC WAS .847 CHECK IF BETTER ACC IF FEATURE SCALING HAPPENS AFTER TRAIN TEST SPLIT

print(a_s) 
# ACC WAS .847 CHECK IF BETTER ACC IF FEATURE SCALING HAPPENS AFTER TRAIN TEST SPLIT
# ACC WAS .850 WHEN FEATURE SCALING HAPPENED AFTER TRAIN TEST SPLIT

    
    






            
### REPORT TO CLIENT

# Dear Sirs please find requested model labelled 'classifier' above. From result of performance evaluation, accuracy of model is 85%. Regards.
    
    
    
    
    





    
    
    


### ### NOTES ### ###
    
    # the human brain
    
        # idea behind deep learning is to mimic and recreate human brain. why?
        
        # because human brain is best known learning machine
        
        # leverage the power of natural selection
        
        # in the human brain you have about 100 billion neurons, each connected to about 1000 of its neighbours
        
        # in deep learning we replicate the brain by creating an artificial neural network
        
    # in a neural network you have three layers
    
        # input layer of neurons (nodes)

        # hidden layers of nodes

        # output layer of nodes

    # each node in each layer is connected to every node in the preceding layer and to every node in the succeeding layer
    
    # in the human brain you have about 100 billion neurons
    
    # each neuron is made up of
    
        # a body
        
        # dendrites which recieve signals
        
        # an axon which transmits signals to the dendrites of other neurons
        
    # synapses
    
        # when an axon transmits a signal a ocnnection is made between the axon of the sending neuron and the dendrite of the receiving neuron
        
        # the connection is called a synapse
        
        # therefore the connections between our nodes will be called synapses
        
        # in the brain neuro-transmitter molecules are sent across the synapse from the sending axon to the receiving dendrite        
    
        # in the neural network each synapse is assigned a weight
        
        # by adjusting the weights the network determines which synapse is important and which is not, including what signal does not get passed along
        
        # gradient descent, back propagation
        
    # independent variables
    
        # when a specific node receives input from the preceding input layer, it receives input from each input node as an independent variable
        
        # ie if there are three nodes in the preceding input layer, then our  receiving node will receive three independent variables
        
        # you should also standardise or normalise the independent variables. why?
        
        # because it will be easier for the neural network to process them if all the values are roughly in the same range of magnitude

        # standardise ie they should have a mean of zero adn a variance of 1
        
        # normalise ie (nth value - min value) / (max value - min value)
        
    # the output layer of a neural network can yield either of
    
        # a continuous output
        
        # a binary output
        
        # a categorical output
        
        # the sigmoid function (see 'sigmoid function') is very frequently used in the output layer, ie the sigmoid function brings every output to between 0 and 1
        
        # using the sigmoid function for the output layer helps us to compute the probability of the outcome belonging to any of the predefined categories

    
    # in summary the nodes of the input layer constitute a single observation, the hidden layer(s) correspond to the same observation and the output layer corresponds to the same observation; in essence a semblance of a multiple linear regression
    
    
    # what happens when a node receives a set of inputs?
    
        # it calculates the sum of the weighted inputs
        
        # it applies an activation function
        
        # it passes on (to the next neuron) the result, if any.
        
        
        
    # activation functions
    
        # threshold function
        
        # sigmoid function
        
        # rectifier function
        
        # hyperbolic tangent function



    # threshold function, 0 <= y <= 1
    
        # y = 0 if x < 0
        
        # y = 1 if x >= 0
        
        
        
    # sigmoid function, 0 <= y <= 1

        # very frequently used in the output layer, ie brings every output to between 0 and 1
        
        # y = 1 / (1 + e^-x)
    
        # very useful in output layer esp. for predicting probability
        
        # advantage over threshold - no kinks
        
        # Probability(sigmoid function = 1) will always either be 0 or 1 to the nearest integer, and therefore the sigmoid function can be used to calculate binary outputs
        
        
        
    # rectifier function, 0 <= y <= x
    
        # very frequently used in the hidden layers
        
        # y = max(x, 0) 
        
        # y = 0 if x < 0
        
        # y = mx if x >= 0 (uniformly increases starting from x = 0)
        
        
        
    # hyperbolic tangent function, -1 <= y <= 1
    
        # y = (1 - e^-2x) / (1 + e^-2x)
        
        
        
    # property valuation task
    
        # a nn takes in some parameters on a property and then values it
        
        # steps in the task
        
            # train a network
            
            # apply it to a task
            
        # typical inputs provided by the input layer
        
            # total flooor area of property, x1
            
            # distance to the city, x2
            
            # age of property, x3
            
            # no. of bedrooms, x4
            
        # assuming there are no hidden layers, the price (output layer) can be calculated simply as
        
            # y = w1(x1) + w2(x2) + w3(x3) + w4(x4)
            
            # where w is the weight for each input
            
        # however a hidden layer gives more detail to our computation, and therefore more power and accuracy to our computation
        
        # in the hidden layer we have neurons and each neuron in the hidden layer receives a signal from each neuron in the input layer, summed up as
        
            # y = w1(x1) + w2(x2) + w3(x3) + w4(x4)

            # where x represents each synapse coming from the input layer and w represents the weight of each synapse
            
            
            
	# selective weighting
		
		# as a consequence of the training of the neural network, each neuron in the hidden layer accepts only some of the inputs from the input layer, and sets the weights of all the other inputs to 0
		
		# the network will give the greatest weight to the inputs (independent variables) that have themost impact on the dependent variable
        
        # it is in fact possible for a node to pick up a selection of inputs that we would never have thought of
    
        # this way, with each neuron focusing only on a specific set of nodes from the preceding layer, it is possible to very highly calibrate the sensitivity/flexibilty of a neural network 
    
        # working in concert all the neurons are able to synergise their specific findings to give you very powerful results
    
    
    
    # single layer feed forward neural network
    
        # also known as a perceptron
        
        # we forward propagate to calculate y^
        
        # then backpropagate to train the network by adjusting the weights
    
        # created in 1957 by Frank Rosenblatt

        # can learn and adjust itself
        
        # y, actual value
        
        # y^ = sum(wi(xi)), estimated value
        
        # cost function, also known as mean-squared-error function, C = 0.5(y^ - y)^2
        
        
        
    # back propagation: loop through the following steps:
    
        # the receiving node receives the inputs from the sending nodes
    
        # calculate y^
        
        # calculate C
        
        # if C is minimised end loop
        
        # if C not minimised back propagate C to the sending nodes
        
        # update the weights with a view to minimizing C
        
        
        
    # epoch
    
        # when the whole training set passes through the ann
        
        # back propagation for the epoch
        
            # for each observation in the dataset calculate the following
            
                # y, actual value
                
                # y^ = sum(wi(xi)), estimated value
                
            # then the cost function is, C = summation(0.5(y^ - y)^2)
            
            # based on the cost function obtained, update each weight w1, w2, w3,... wn, and recalculate the cost function
            
            # remember that the same set of values for w1, w2, w3,... wn is applied to each observation in the dataset (confirm this please)
            
            # repeat the process until the cost function is minimised
            
            
            
    # batch gradient descent
    
        # alternative for attempting to minimise cost function by trying out an unlimited number of values for a weight
        
        # this approach will eventually succeed in minimising C but as your synapses increase in number you encounter the curse of dimensionality
        
        # curse of dimensionality: so many dimensions it would take nearly all of eternity to rocess all combinations for a large network
        
        # gradient descent maens that for each value of y^ check the angle of the gradient on the curve of C vs y^
        
            # go right if gradient is negative, or go left if gradient is positive
            
            # repeat until gradient = 0
            
        # adv: deterministic
        
        # if you have the same starting weights you'll have the same results (iterative process) for the way your weights are updated



    # stochastic gradient descent (reinforcement learning)
    
        # required for a cost function that is not smoothly convex (for which a simple batch gradient descent would apply) but instead is very uneven

        # a simple g descent is very likely to give us a local min and the global min
        
        # here we find y^ for each ob one at a time, and we calculate C AND adjust the weights each time before moving on to the next ob
        
        # it is faster than batch gradient descent 

        # even if you have the same starting weights you might not get the same results
        
        
        
    # mini batch gradient descent
    
        # cross btw batch gradient descent and stochastic
        
        # calculate C for a few obs at a time, and adjust weights each time
        
        
    
    # back propagation
    
        # all weights are adjusted simultaneously

        # you know what part of the error each of your weights is responsible for
        
    

    
    ### uses of deep learning
            
        # making predictions for business problems, ann
        
        # making classifications for business problems, ann
        
        # computer vision, cnn
        
            # recognising faces in pictures and videos
            
            # recognising tumors in medical images
            
        # recommender systems, deep boltzmann machines
    
    

    
    # theano
    
        # open source numerical computations library
        
        # based on numpy syntax
        
        # runs on cpu and gpu
        
    
    
    
    # tensorflow
    
        # open source numerical computations library
        
        # developed by google
        
        # runs on cpu and gpu
         
        
    
    
    # gpu is better for deep neural networks
    
        # gpu has more cores
        
        # can run more fpc's per second than the cpu
        
        # better for computing intensive tasks and parallel computations
        
        
        
        
    # keras
    
        # library based on theano and tensorflow
        
        # developed by google scientist
        
        

