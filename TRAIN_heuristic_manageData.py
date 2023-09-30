# INDEX
# get_arrays_of_indices:
#   e.g.  INP:  num_nodes = 3 ->  OUT:    [0,1,2]
# extract_data:
#   e.g.  INP:  name_of_files ->    OUT:   problem_data
# inizialise solution variables:
#   e.g. INP:  number_of_nodes ecc..   ->  OUT: empy_array for x,w,y,z (decision variables)



# Lib
import random
import numpy as np
import gurobipy as gb
import matplotlib.pyplot as plt
import random
import networkx as nx 
import math
from itertools import chain
import networkx as nx
import re

# Our libs
import TRAIN_heuristic_plotFunctions
import TRAIN_heuristic_randomSolution



# Definition of indices
def get_arrays_of_indices(num_nodes, num_terminals, num_costumers, num_trailers):
    trailer_indices = [(i + 1) for i in range( num_trailers )]
    terminals_indices = [(i + 1) for i in range( num_terminals ) ]
    costumers_indices = [ ( i + 1 + num_terminals ) for i in range( num_costumers )  ] 
    nodes_indices = [ i for i in range( num_nodes + 1 ) ]

    return nodes_indices, terminals_indices, costumers_indices, trailer_indices




# given the filename of the original instance in the dataset, it returns the wanted variables for initializing the model
def extract_data(filename):
    LTL_orders = int(filename[7])
    LTL_patter = filename[9]
    filename = "./dataset/" + filename
    FILE = open(filename, 'r')
    lines = FILE.read().strip().split("\n")
    input_data = [list(map(int, re.findall(r'\d+', line))) for line in lines]
    num_costumers =             input_data[0][0]
    num_terminals =             input_data[0][1]
    Q =                         input_data[0][2]
    c1 =                        input_data[0][3]/10
    c2 =                        input_data[0][4]/10
    num_experienced_drivers =   input_data[0][5]
    costumers_data = np.array([row[1:4] for row in input_data[1: num_costumers + 1]])
    #print(costumers_data)
    terminals_data = np.array([row[1:3] for row in input_data[num_costumers + 1: num_costumers + num_terminals + 2]])
    #print(terminals_data)
    terminals_data = np.hstack((terminals_data, np.zeros((num_terminals,1))))
    dummy_data = np.array([[0, 0, 0]])
    nodes = np.vstack((dummy_data, terminals_data, costumers_data))
    #print(nodes)
    q = [0,0,0,0] + [row[3] for row in input_data[1: num_costumers + 1]]
    FILE.close()
    return  num_costumers, num_terminals, Q, c1, c2, num_experienced_drivers, nodes, q, LTL_orders, LTL_patter





# Initialise solution variables
def solutionVariables_initialization(num_nodes, num_terminals, num_costumers, num_trailers ):
    y = np.zeros( (num_terminals + 1, num_trailers + 1) )
    z = np.zeros( (num_terminals + 1, num_trailers + 1, num_costumers + num_terminals + 1) )
    w = np.zeros( num_terminals + 1 )
    x = np.zeros( (num_nodes + 1 , num_nodes + 1, num_terminals + 1, num_trailers + 1) )
    return y,z,w,x
