### ### PROBLEM






### ### PLAN





    
### ### IMPLEMENTATION










### ### NOTES

#reputation
    #the most used unsupervised
    #the most popular dr algo


#aim
    #identify strength of correlation between variables
    #reduce number of independent variables to 2
    

#objective
    #ml algorithm is likely to have greater accuracy of predictions
    #with 2 or 3 independent variables it possible to visualize, in 2d or 3d, the relationships between the variables


#method
    #find the directions of maximum variance in high dimensional data
    #retain most of the information
    #project the variance into a smaller dimensional sub-space


#general outcomes
    #visualisation
    #noise filteirng
    #feature extracftion


#real world applications
    #stock market prediction
    #gene analysis


#steps
    #standardise
    #obtain eigenvalues and eigenvectors
    #sort eigenvalues in descending order, choose the eigenvectors that correspond to the k largest eigenvalues where k is the size of the new feature sub space
    #construct the projection matrix from the eigenvectors
    #using the projection matrix, transform the original feature dataset to the new feature dataset
        
    
#steps summarised
    #find the relationships between x and y
    #list the principal axes
    

#weakness
    #affected by outliers