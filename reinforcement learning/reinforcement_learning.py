# source: manuel amunategui https://www.youtube.com/watch?v=nSxaG_Kjw_w

# map
# goal
# reward matrix
# learning matrix and learning parameter
# brain
# train
# test



import numpy as np
import pylab as plt


### ### PLAN ### ###



# DEFINE MAP
    # create points list
    # import networkx
    # create networkx graph object
    # add edges from points list
    # create dictionary of coordinates for layout
    # draw nodes, edges, labels


# DEFINE GOAL
    # define goal
        

# DEFINE REWARD MATRIX
    # initialise reward matrix
    # install rewards and zeros in reward matrix
    

# DEFINE LEARNING MATRIX AND LEARNING PARAMETER 
    # define learning matrix
    # define learning paramater


# DEFINE BRAIN
    # define function to create list of available actions
    # define function to randomly select one of the available actions
    # define function to update learning matrix (ie with performance score of action selected) based on action selected, corresponding score from reward matrix, and learning parameter


# TRAIN


# TEST








### ### IMPLEMENTATION ### ###



### DEFINE MAP ###

	
	### map cell to cell, add circular cell to goal point

    
    # create points list
points_list = [(0,4), (4,8), (4,7), (4,11), (5,78), (6,8), (6,1), (1,23), (1,12), (6,17), (17,18), (17,25), (0,9), (78,10), (9,5), (9,19), (9,20), (5,15), (5,16), (78, 75), (78,76)]



	# import networkx
import networkx as nx


	# create networkx graph object
graph = nx.Graph()


    # add_edges_from points list
graph.add_edges_from(points_list)


    # create dictionary of coordinates for layout
positions = nx.spring_layout(graph)


	# draw nodes, edges, labels
nx.draw_networkx_nodes(graph, positions)
nx.draw_networkx_edges(graph, positions)
nx.draw_networkx_labels(graph, positions)
plt.show()






### DEFINE GOAL ###

goal = 78

# from structure of points_list, we can see that ideal route from origin to goal is 0,4,5,78,10. all other points are noise







### DEFINE REWARD MATRIX ###



### INITIALISE REWARD MATRIX IN A FORM THAT THE Q LEARNING ALGO CAN READ

MATRIX_SIZE = 79

R = np.matrix(np.ones(shape = (MATRIX_SIZE, MATRIX_SIZE)))

R = R * -1

#print(R)



### INSTALL REWARDS AND ZEROS IN REWARD MATRIX

# the objective is that:
    # points where the algorithm shouldn't visit are left labelled -1
    # points where the algorithm can visit but that have no reward are labelled 0
    # points where the algorithm can visit and that have reward are labelled 100
    # note: the reward matrix is not shown to the algorithm. The algorithm uses it after the fact to check if it made a point or lost a point
    

for point in points_list:
    
    if point[1] == goal: 
        
        R[point] = 100
        
    else:
        
        R[point] = 0
        
        
        
    if point[0] == goal:
        
        R[point[::-1]] = 100
    
    else:
        
        R[point[::-1]] = 0


# create a redundancy which rewards the algorithm when it moves from the goal to the goal. It is redundant but the algorithm needs it

R[goal, goal] = 100

#print(R)







### DEFINE LEARNING MATRIX AND LEARNING PARAMETER ### 


### DEFINE THE LEARNING MATRIX

# this is where the model will save how well its done on its different routes
# it will keep a tally of where it does well
# the biggest numbers will be the most efficient route

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

#print(Q)




### DEFINE LEARNING PARAMETER (GAMMA)

gamma = 0.8

#print(gamma)






### DEFINE BRAIN ###




# DEFINE FUNCTION TO CREATE LIST OF AVAILABLE OPTIONS

def available_options(current_position): # this function returns every potential next move, starting from the algorithms current position
    
    current_row = R[current_position,]  # return row corresponding to the current state

    
    available_moves = np.where(current_row >= 0)[1]
    
    
    return available_moves # returns all available moves, in a list







# DEFINE FUNCTION TO RANDOMLY SELECT ONE OF THE AVAILABLE ACTIONS

def choose_next_action(options): # this function randomly picks one of the available moves
    
    
    next_action = int(np.random.choice(a = options, size = 1)) # randomly chooses from options_list, always remember the size attribute
    

    return next_action







# DEFINE A FUNCTION TO UPDATE LEARNING MATRIX (IE WITH THE PERFORMANCE SCORE OF ACTION SELECTED) BASED ON ACTION SELECTED, CORRESPONDING SCORE FROM REWARD MATRIX, AND LEARNING PARAMETER

# this function will save how well the model has done on its different routes
# this function will keep a tally of where the model does well
# the biggest numbers will be the most efficient route
# it does this by looking at where the bot went and then looking at the reward matrix and seeing the score it got for going there

def update(current_position, action, gamma): # this function is the brain; in this function you can tune the parameter labelled 'gamma' - the learning parameter
    
    # define the max_index

    max_index = np.where(Q[action, ] == np.max(Q[action,]))[1] 
    
    


    # update the max_index

    if max_index.shape[0] > 1:
        
        max_index = int(np.random.choice(max_index, size = 1))
    
    else:
    
        max_index = int(max_index)
    
    


    # define the max_value
    
    max_value = Q[action, max_index]
    
    


    # update the Q matrix (learning matrix) using the current state, the action taken, gamma and, the max_value
    
    Q[current_position, action] = R[current_position, action] + gamma * max_value
    


    
    #print('max_value', R[current_position, action]  + gamma * max_value)
    
    


    # return a score that indicates how successful the action taken was
    
    if (np.max(Q) > 0):
        
        return (np.sum(Q/np.max(Q) * 100))
    
    else:
    
        return (0)
    






### TRAIN ###

    # define list for scores
scores = []



    # run the update function x times to allow the model to figure out the most efficient path
    # each time, calculate the performance score and append it to the score list

for i in range(1000000):
    
    my_position = np.random.randint(0, int(Q.shape[0]))
    
    #print('current position is ', my_position)
    
    
    my_options_list = available_options(my_position)
    
    #print('options list is ', my_options_list)
    
    
    if len(my_options_list) == 0:
        
        continue

    else:
        
        my_action = choose_next_action(my_options_list)
        
        #print('selected action is ', my_action)
        
    
        score = update(my_position, my_action, gamma)
        
        #print('score is ', score)
        
    
        scores.append(score)
        
        #print('scores are ', scores, '\n')
        
        #print('Score: ', str(score))
    
    
    

### TEST ###
    
current_state = 18

steps = [current_state]

while current_state != 78:
    
    
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state,]))[1] 



    if next_step_index.shape[0] > 1:
    
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    
    else:
        
        next_step_index = int(next_step_index)
    
    
    steps.append(next_step_index)

    
    current_state = next_step_index
    

print('most efficient path: ')

print(steps)


#plt.plot(scores)

#plt.show()






