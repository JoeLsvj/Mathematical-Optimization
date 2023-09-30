# Lib
import random
import numpy as np
import gurobipy as gb
import matplotlib.pyplot as plt
import random
import networkx as nx 
import math
from itertools import chain

# Our libs
import TRAIN_heuristic_plotFunctions

# Build a random solution

def random_solution(nodes_matrix, num_terminals, num_trailers, trailer_capacity, number_of_experienced_drivers):

    # Indices variable
    num_nodes = len(nodes_matrix) # e.g. {0,1,2} num_nodes = 3
    indices_of_nodes = [i for i in range(num_nodes)] # e.g. {0,1,2} nodes_indeces = [0,1,2]

    J = indices_of_nodes[num_terminals + 1:] # costumers indices
    K = [i for i in range(num_trailers + 1)] # trailers indices
    T = indices_of_nodes[1: num_terminals + 1] # terminal indices

    # Solution variables
    y = np.zeros( (num_terminals + 1, num_trailers + 1) )
    z = np.zeros( (num_terminals + 1, num_trailers + 1, num_nodes) )
    w = np.zeros( num_terminals + 1 )
    x = np.zeros( (num_nodes, num_nodes, num_terminals + 1, num_trailers + 1) )

    # Trailer data:  [ id , good current availability , current position ]
    trailer_data = [[k,trailer_capacity,1] for k in K] # (id, current capacity, node_position)

    # Trailer place in road-train
    # Send road-train to terminal t -> 2 trailers you can send to terminal t 
    # If trailer_place_in_road_train_to_terminal[t] = 0 means if you want to send k to t you need a further road-train
    # If trailer_place_in_road_train_to_terminal[t] = 1 means if you want to send k to t you can just pack k in a previous sent road-train
    trailer_place_in_road_train_to_terminal = np.zeros( num_terminals + 1 )

    # Execute while_1.
    # When a costumer is not served during while_1, it gets in in J_bar 
    J_bar = [] # J_bar = Clients still not served AND unserviceable with k from t 

    # While_1
    # Fix a terminal t, a trailer k. 
    # Serve more clients you can with trailer k from terminal t.
    while len(J) != 0: 

        k = random.choice( K[1:] ) # NB trailer 0 does not exist
        t = random.choice( T ) 

        # While_2
        # Fix a cosumer j
        # Serve j only if the current capacity of k is greater or equal then the request.
        while len(J) != 0: 
            j = random.choice(J)

            # Choosen terminal t is secondary terminal -> Need of road-train
            # No more drivers ->  Not choose t as assigned terminal
            drivers_availability = (np.sum(w) < number_of_experienced_drivers )

            # nodes_matrix[j][2] : request of costumer j
            # trailer_data[k][1]: current available goods of trailer k
            if nodes_matrix[j][2] <= trailer_data[k][1]:

                trailer_position = trailer_data[k][2]
                if trailer_position != 1: # Path client-client
                    if trailer_position < j: # Just to store x with first_argument < second_argument
                        x[trailer_position, j, t, k] += 1
                    else:
                        x[j, trailer_position, t, k] += 1
                    trailer_data[k][2] = j
                    # Serve client j SO remove availability of trailer k
                    trailer_data[k][1] -= nodes_matrix[j][2]
                    y[t][k] = 1 # upload solution
                    z[t][k][j] = 1
                elif trailer_position == 1:
                    if t == 1: # Path original_terminal-client 
                        x[1,j,t,k] += 1 
                        trailer_data[k][2] = j
                        # Serve client j SO remove availability of trailer k
                        trailer_data[k][1] -= nodes_matrix[j][2]
                        y[t][k] = 1 # upload solution
                        z[t][k][j] = 1
                    elif trailer_place_in_road_train_to_terminal[t] == 1: # you can pack this trailer in a previous road-train to sec_terminal
                        x[t,j,t,k] += 1 # Path sec_terminal-client 
                        trailer_data[k][2] = j
                        trailer_place_in_road_train_to_terminal[t] -= 1
                        # Serve client j SO remove availability of trailer k
                        trailer_data[k][1] -= nodes_matrix[j][2]
                        y[t][k] = 1 # upload solution
                        z[t][k][j] = 1
                    elif drivers_availability: # Path original_terminal-secondary_terminal + Path sec_terminal-client  
                        w[t] += 1
                        trailer_place_in_road_train_to_terminal[t] += 1
                        x[t,j,t,k] += 1 # Path sec_terminal-client 
                        trailer_data[k][2] = j
                        # Serve client j SO remove availability of trailer k
                        trailer_data[k][1] -= nodes_matrix[j][2]
                        y[t][k] = 1 # upload solution
                        z[t][k][j] = 1
                    else: # Assigned terminal is not the original, BUT there aren't drivers
                        J_bar = J_bar + [j] # J_bar = Clients still not served AND unserviceable with k from t

                # J is served -> remove it permanently 
                J.remove(j)

            else:
                # j is not served -> Move j from J to J_bar
                J.remove(j)
                J_bar = J_bar + [j] # J_bar = Clients still not served AND unserviceable with k from t

        # End of while_2 
        K.remove(k)
        
        J = J_bar.copy()
        J_bar = []
    
    # Come back violento 
    # Each clients now is satisfied. 
    # If a trailer does not stay in a terminal then make it come back to the respective terminal
    terminal_indices = indices_of_nodes[1: num_terminals + 1]
    for k in range(num_trailers): # for each trailer k
        k += 1
        trailer_position = trailer_data[k][2] # find k position
        for t in terminal_indices: # find terminal associated to k
            if y[t][k] == 1:
                associated_terminal = t
                break
            else:
                associated_terminal = 0
        if not (trailer_position in terminal_indices): # If k is not in a terminal makes it come home
            x[associated_terminal, trailer_position, associated_terminal, k] += 1 

    # While_1 end


    # Correct situation like: a road train is sent with only one trailer
    J = indices_of_nodes[num_terminals + 1:] # costumers indices
    K = [i for i in range(num_trailers + 1)] # trailers indices
    T = indices_of_nodes[1: num_terminals + 1] # terminal indices

    for t in [2,3]:
        costumer_removed = []
        if trailer_place_in_road_train_to_terminal[t] == 1:
            # Remove the road train to t
            w[t] -= 1
            # take a trailer k assigned to t
            for k in K:
                if y[t][k] == 1:
                    assigned_trailer = k
                    break
            # reset all the route of k from t
            y[t][assigned_trailer] = 0
            for j in J:
                if z[t][assigned_trailer][j] == 1:
                    costumer_removed.append(j) 
                    z[t][assigned_trailer][j] = 0 
            for i in costumer_removed:
                x[t][i][t][assigned_trailer] = 0
                for j in costumer_removed:
                    if i < j:
                        x[i][j][t][assigned_trailer] = 0
        
            # build a route for k from t = 1
            y[1,assigned_trailer] = 1
            for j in costumer_removed:
                z[1][assigned_trailer][j] = 1 
            if len(costumer_removed) == 1:
                j = random.choice(costumer_removed)
                x[1][j][1][assigned_trailer] = 2
            elif len(costumer_removed) > 1:
                i = random.choice(costumer_removed)
                x[1][i][1][assigned_trailer] += 1
                costumer_removed.remove(i)
                while(len(costumer_removed) > 0):
                    j = random.choice(costumer_removed)
                    if i<j:
                        x[i][j][1][assigned_trailer] += 1 
                    else:
                        x[j][i][1][assigned_trailer] += 1 
                    costumer_removed.remove(j)
                    i = j
                x[1][i][1][assigned_trailer] += 1

    return y,z,x,w
