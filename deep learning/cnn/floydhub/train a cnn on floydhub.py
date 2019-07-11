# an ann is a good classifier for non-linear problems



### ### PROBLEM ### ###

    # please state problem here
    






### ### PLAN ### ###

### find
    
### explore
    
### prepare
    
### analyse
    
    # build classifier

        # create object

        # add convolution layer
        
        # add pooling layer
        
        # add flattening layer
        
        # perform full connection

            # add layers
            # compile
            
    # fit classifier to dataset
    
    
### how do we display the accuracy of the model?

### how do we take a random image as input, analyse it using the classifier and classify it as cat or dog?











### ### IMPLEMENTATION ### ###

### find
    
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense





### explore



    
### prepare



    

### analyse
    


    # build classifier



        # create object

classifier = Sequential()


        

        # add convolution layer

# deprecated? classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# filters = 32, if your PC is running your cnn using a cpu and not a gpu, the recommendation is that you start with filters=32 so that you can test the capacity of your PC gradually (beginning with small loads)
    # as time goes on you may add on other convolutional layers with 64 or 128 filters

# border_mode (padding?) - implies how the filter will handle the borders of the input image

# input_shape - will convert all our images to the same format input_shape = (rows, columns, channels)
    # coloured images will be converted to a 3D array during image preprocessing; the array consists of 3 channels - rgb
    # greyscale images will be converted to a 2D array during image preprocessing; the array consists of 1 channel - black(?)
    # each channel consists of a 2d array that contains the pixels of the image
    # 256x256 are common dimensions of the 2D array in each of the channels (recommendation: use 64x64 in order not to overload your PC)
    # the order of the arguments in input_shape are reversed depending on if you're using theano (channels first) or tensorflow (channels last)

# rectification
    # activation = rectified linear unit
        
        
        


        # add pooling layer

classifier.add(MaxPooling2D(pool_size = (2,2)))
        
        



        # add flattening layer

classifier.add(Flatten())
        
        




        # perform full connection


            # add hidden layer

classifier.add(Dense(units = 128, activation = 'relu'))
# deprecated 'classifier.add(Dense(output_dim = 128, activation = 'relu'))'
# output_dim - number of nodes in the hidden layer
    # no rule of thumb for this
    # common practice - choose number between number of input and number of output nodes
    # choice often comes from experimentation
    # pick a number not too small and not too big
    # pick a power of 2 eg 128
# activation - relu


            # add output layer

classifier.add(Dense(units = 1, activation = 'sigmoid'))
# deprecated classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
# activation used here is sigmoid because outcome is binary
# aim of activation here is to predict probability of a number of outcomes
# if outcome can be placed in more than two categories we would need to use the soft max activation function



            # compile - Configures the model for training

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# optimizer = 'adam' which implies stochastic gradient algorithm
# loss = loss function; depends on number of categories of outcome
# metrics = performance metric
    
            




    # fit classifier to dataset

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#training_set = train_datagen.flow_from_directory(directory='gboyegaanu/datasets/set_training/1:training',
#training_set = train_datagen.flow_from_directory(directory='https://www.floydhub.com/gboyegaanu/datasets/set_training/1:training',
#training_set = train_datagen.flow_from_directory(directory='datasets/set_training/1:training',
#training_set = train_datagen.flow_from_directory(directory='set_training/1:training',

#training_set = train_datagen.flow_from_directory(directory='gboyegaanu/datasets/set_training/1',
#training_set = train_datagen.flow_from_directory(directory='https://www.floydhub.com/gboyegaanu/datasets/set_training/1',
#training_set = train_datagen.flow_from_directory(directory='datasets/set_training/1',
#training_set = train_datagen.flow_from_directory(directory='set_training/1',

#training_set = train_datagen.flow_from_directory(directory='gboyegaanu/datasets/set_training',
#training_set = train_datagen.flow_from_directory(directory='https://www.floydhub.com/gboyegaanu/datasets/set_training',
#training_set = train_datagen.flow_from_directory(directory='datasets/set_training',
#training_set = train_datagen.flow_from_directory(directory='set_training',

#training_set = train_datagen.flow_from_directory(directory='gboyegaanu/set_training',
#training_set = train_datagen.flow_from_directory(directory='https://www.floydhub.com/gboyegaanu/set_training',

#training_set = train_datagen.flow_from_directory(directory='gboyegaanu/datasets/set_training/1/training',

#training_set = train_datagen.flow_from_directory(directory='gboyegaanu/datasets/set_training/training',

training_set = train_datagen.flow_from_directory(
        directory='/floyd/input/training',
        target_size=(64, 64),
        batch_size=32, # NUMBER OF IMAGES PROCESSED BEFORE GRADIENT UPDATE?
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        directory='/floyd/input/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# use keras documentation for image augmentation
# image augmentation - preprocessing your images (to enrich your dataset) to prevent overfitting (great results on training set and poor result on test set)
# for this we will use the imageDataGenerator from keras documentation
# overfitting usually happens if our input data is too small
# it isnt sufficient that the cnn finds correlation between some independent variables and some dependent variables, rather the cnn needs to find patterns in the pixels
# therefore we either need more images or a trick: image augmentation
# divide all your images into batches and in each batch apply random transformation/distortions (flip, rotate, shear, shrink) to our images and thereby create a virtual pool of images that is larger than the actual pool of images
# options:
    # flow
    # flow from directory
# shear range - transvection

# for greataer accuracy try target_size=(128,128)
# for target_size=(128, 128) you're better off processing with a gpu rather than a cpu otherwise processing will take a really long time


    



"""


# OPTIONS

# You can iterate on your training data in batches:
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
classifier.fit(x_train, y_train, epochs=5, batch_size=32)




# Alternatively, you can feed batches to your model manually:
classifier.train_on_batch(x_batch, y_batch)




# Evaluate your performance in one line:
loss_and_metrics = classifier.evaluate(x_test, y_test, batch_size=128)



# Or generate predictions on new data:
predicted_classes = classifier.predict(x_test, batch_size=128)






# OPTION SELECTED

# generate predictions on new data:

predicted_class = classifier.predict('/floyd/input/test/cats/cat.4001.jpg')
print(predicted_class)


predicted_class = classifier.predict('/floyd/input/test/dogs/dog.4001.jpg')
print(predicted_class)







# INVESTIGATE THE FOLLOWING (SEE https://keras.io/models/sequential/#predict)


predict

predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)

Generates output predictions for the input samples

Computation is done in batches







predict_on_batch

predict_on_batch(x)

Returns predictions for a single batch of samples.

Arguments:

x: Input samples, as a Numpy array.

Returns:

Numpy array(s) of predictions.







evaluate_generator

evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

Evaluates the model on a data generator.

The generator should return the same kind of data as accepted by test_on_batch.







predict_generator

predict_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

Generates predictions for the input samples from a data generator.

The generator should return the same kind of data (A NUMPY ARRAY?) as accepted by predict_on_batch.


"""









### ### NOTES ### ###
    
    
    # proponents
    
        # geoffrey hinton - godfather of ann, boss at google
        
        # yann lecun - student of geoffrey hinton, godfather of cnn, boss at facebook
    
    
    
    # biological basis
    
        # we see images because our brain is looking for and is processing visual features
        
        # our brain classifies each image based on the features it has seen and processed
        
        # sometimes our brain classifies an image wrongly (and later realise that what we thought we saw was actually something else) possibly because our brains did not have enough time to process the features we saw
        
        
    
    # main topics
        
        # convolution operation
        # relu layer
        # pooling 
        # flattening
        # full connection
        # softmax and cross entropy
        
        # sub sampling is a generalizaion of mean pooling
        
        
    
    # steps in cnn process
        
        # convolution - create convolutional layer using feature detector
        # rectification - remove linearity using rectifier and create a relu layer
        # pooling - create pooled feature maps 
        # flattening
        # full connection
    


    # step 1a convolution - to create a convolutional layer that consists of several feature maps
    
        # input image - an x by x matrix of 1s and 0s
        
        # feature detector, or kernel or filter - a y by y (3 by 3 is common) matrix of 1s and 0s
    
        # the convolution is the integration of the feature detector and the input image

        # superimpose the feature detector on the input image
                
        # take the product of the integers in the corresponding cells
        
        # sum the products
        
        # write the sum in the first cell of a matrix (the feature map)
        
        # advance the feature detector by the stride you have defined (eg a one cell stride (2 is common)) and repeat the process
        
        # the resulting feature map is known as a convolved feature or activation map
        
        # the feature map achieves several things
        
            # reduces the size of the image making it easier to process (the larger the stride, the smaller the feature map)
            
            # it detects parts of the image that are integral (even though some information is lost)
            
                # the feature detector has a pattern on it
                
                # the highest numbers on the feature map indicate where the pattern finds a match on the input image
            
            # it preserves the relationships btw the images
            
            # most of the time the features the machine detects might mean nothing to a human eye
                
        # the feature map proves that we dont consider every single detail of every thing we see, instead we look for key features
        
        # the feature map helps us to bring forward and preserve these key features
        
        # we repeat the process with several feature detectors and thereby create multiple feature maps to obtain our first convolutional layer
        
        # thru its training the cnn decides which features are important for certain categories of images and it has filters for finding those features
    
        # see docs.gimp.org for downloadable feature detectors with which you can modify your personal photos and images by applying the feature detectors to the images
        

        
        
    
    
    # step 1b rectification - rectified linear unit
    
        # apply a rectifier to the convolutional layer
        
            # changes all negative input to 0??
    
            # done to increase non-linearity in our convolution
        
        # when applying convolution we risk that we might create someting linear

        # the rectifier breaks up linearity
        
        # for example a convolved image might feature an unwanted progression from white to grey to black
        
        # the rectifier eliminates the black and thereby breaks the linearity
        
        # why apply a rectifier? because images are highly non linear - many non linear elements and transitions
        
        # a non linear activation function is essential at the filter output of all intermediate layers
        
        # see also the parametric relu described by Kaiming et al at https://arxiv.org/pdf/1502.01852.pdf
        
    
    
    
    # step 2 pooling 
    
        # if a cnn seeks to classify a visual entity by locating a distinguishing feature with exactly the same properties (position, shape, texture) in every image where the entity is to be identified, the cnn might not be successful 100% of the time

		# this is because the position, shape or texture of the feature might vary from  one image to another

		# every cnn must therefore have spatial invariance; flexibility 
        
        # ie even if a feature is slightly distorted the cnn must be able to find it
        
        # this is the aim of pooling
        
        # there are a few types of pooling: max pooling, mean pooling, sum pooling, etc
        
        # pooling is carried out after feature mapping is completed
        
        # steps
        
            # consider a 2x2 sample in the top left hand corner of the feature map
            
            # max pooling - find the highest value in that sample and write it down in a new matrix called the pooled feature map
            
            # advance to the right by a stride that you have defined (usually a stride of 2) and repeat the process until you have considered the entire feature map
            
        # advantages
        
            # reduces complexity without loosing performance
            
            # max pooling still preserves the features very accurately
            
            # max pooling gets rid of 75% of less inportant information thereby speeding up processing 
                           
            # by taking the max number in each image we account for any distortion to or rotation to the image being analysed by the cnn, this is because when slight distortions occur, the max values still end up in the same 
            
            # reduces the unmber of parameters and thereby prevents overfitting onto irrelevant information   
        
        # why pooling? 
    
            # to reduce the number of nodes
            
            # thereby reducing complexity and processing time
        
            # so that network is less compute intensive 
    
    
    
    
    # useful functions - soft max and cross entropy
    
        # the soft max fuunction is used for making all the probabilities in the output layer add up to 1?
        
        # cross entropy - a loss function
        
        # we aim to minimise the cross entropy in order to optimise the performance of our network
    
        # other errors 
        
            # classification error - ie regardless of the probability was the classification right or wrong?
            
                # not sensitive enough    
            
                # not very good for back propagation
                
            # mean squared error - generally more sensitive than the classification error
        
        # why cross entropy instead of mean squared error?
        
            # at the beginning of the back propagation stage, the output value is sometimes very tiny
            
            # smaller than the value you want
            
            # therefore at the start, the gradient in the gradient descent will be very low
            
            # therefore it will be hard for the neural network to make any adjustments to the weights
            
            # whereas because the cross entropy has an logarithm in it, it is able to assess even very small errors
            
            # eg the outcoome you want is 1 and right now you get 1/1000000
            
            # if your outcome improves from 1/1000000 to 1/1000 and you compare the mean squared error for both cases, you'll find that you've not improved your network that much; and this kind of difference will not guide convergence of your network with any significant power
            
            # however if you look at the cross entropy you'll find that you've improved the network significantly; this magnitude of change in the error will drive convergence of your network with greater power and therefore help your network to converge faster
            
    
    
    
    # step 3 flattening
            
        # do this to all the pooled feature maps
        
        # from the pooled feature map, take each record, starting from the top row and moving from left to right
		
        # place each record in a column top to bottom

        # why? because a subsequent step involves putting our data into an ann for further processing   
		
        # the column (vector) therefore serves as an input layer (one single observation?) for our ann 
  
        # in the flattened layer each datapoint represents the spatial relationship of succesive groups of pixels in the input image. This is more useful than having each datapoint represent an individual pixel from the input image
        
            # you are likely to find a selected pixel in almost any image on earth, but you are more likely to find a spatial relationship in only a certain category of images
    
    

    
    # step 4 full connection


		# pass the input layer to a fully connected hidden layer of your ann

		# features are encoded into the numbers in the vector of outputs from the flattening stage
		
		# these features can predict classes 

		# but the purpose of our ann is to combine our features into attributes that predict the classes better
		
		# if the ann predicts a wrong class a cross entropy error function is back propagated through the network
		
		# the weights of the synapses are adjusted via gradient descent 
		
		# the feature detector is adjusted (because we might have an incorrect feature detector looking for the song features) also via gradient descent 
		
		# this continues until the network is optimised
		
		# each neuron in the hidden layer preceding the output layer represents a feature for a specific category of images (eg dog, cat)

		# each neuron in the output layer will give larger weights to the synapses that connect them to relevant neurons in the layer preceding the output layer
		
		# in the hidden layer preceding the output layer the neurons will vote to assign a probability to each neuron (prediction) in the output layer

		# the weights of the synapses connecting them to the output layer indicate the importance of their votes

        
  
    
    # science of cnn's
    
        # math
        
            # a convolution is a mathematical operation on two functions 
            
            # it produces a third function that expresses how the shape of one is modified by the other
            
            # a convolution is a combined integration of 2 functions
            
            # it shows you how one function modifies the shpae of the other
        
        
        # brain
        
            # the convolutions of the brain increase the surface area, or cortex
            
            # THE CONVOLUTIONS ALLOW MORE CAPACITY FOR THE NEURONS that store and process information      
        
        
        # computer science - a cnn takes an image as input and gives us a label (lion, ship, plane, happy, livid, spoon, etc.) as output
        
        
        # to classify a greyscale image a cnn looks at each pixel in the image
        
            # a greyscale image with 4 pixels, 2x2, is received by a cnn as 2d array
            
            # one dimension for the number of columns of pixels and one for the number of rows of pixels
            
            # each datapoint in the array is an integer btw 0 and 255 which represents the intensity (pixel value) of the pixel
            
            # the cnn therefore views the greyscale image as a matrix of integer values
            
            
        # to classify a colored image a cnn looks at each pixel in the image
        
            # a colored image with 4 pixels, 2x2, is received by a cnn as 3d array
            
            # one dimension for the number of columns of pixels, one dimension for the number of rows of pixels and one dimension for the number of arrays (in this case 3 arrays; one each for red, green and blue)
            
            # each datapoint in each array is an integer btw 0 and 255 which represents the intensity (pixel value) of the pixel in the color in question
            
            # the color of the pixel is the combination of colors - at that position - from the r, g and b arrays
            
            # the cnn therefore views the colored image as 3? matrices of integer values





### ### APPENDIX ### ###
            
            
            # troubleshooting installation problems
            # https://github.com/tensorflow/tensorflow/issues/7705
