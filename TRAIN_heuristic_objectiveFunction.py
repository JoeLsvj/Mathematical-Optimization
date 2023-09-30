# INDEX
# compute cost matrix:
#   INP: nodes_matrix, road_costs -> OUT: matrix M s.t. m[i][j] = cost of going from i to j
# exctract_highway_cost:
#   INP: cost_matrix -> OUT: array A where A[i] cost of going from original terminal to a secondary terminal 
# objective_function:
#   INP: decision_variables   ->   OUT: objective_function_value 


# Libraries
import numpy as np
import TRAIN_heuristic_manageData 



# Compute the matrix of costs for the regular trucks, and the vector of costs for the road trains:
# Matrix of distances between each node (terminals and customers included)
def compute_cost_matrix(nodes, c1, c2, num_terminals):
    num_nodes = len(nodes)
    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            x1, y1 = nodes[i][0:2]
            x2, y2 = nodes[j][0:2]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Highway case
            if (i == 0 and j != 0 and j < num_terminals) or (j == 0 and i != 0 and i < num_terminals):
                cost_matrix[i][j] = round(distance * c2, 2)
            # Standard road case
            else:
                cost_matrix[i][j] = round(distance * c1, 2)
    return cost_matrix





# Extract highway costs
def exctract_highway_costs(cost_matrix, num_terminals):
    # list implementation: also the distance original terminal - dummy terminal
    # is included, but not considered later in the gurobi implementation, with the set indices
    highway_cost = cost_matrix[1][0 : num_terminals+1]
    return highway_cost
    # dictionary implementation: indices start from 1 (because secondary terminals are in nodes 1 and 2 in this formulation)
    #highway_cost = road_cost_matrix[0][1 : num_terminals]
    #indices = list(range(1, num_terminals))
    #return dict(zip(indices, highway_cost))




def objective_function(num_nodes, num_terminals, num_costumers, num_trailers, nodes_matrix, c1, c2, x, w):
    nodes_indices, terminals_indices, costumers_indices, trailer_indices = TRAIN_heuristic_manageData.get_arrays_of_indices(
    num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers)
    cost_matrix = compute_cost_matrix(nodes=nodes_matrix, c1=c1, c2=c2, num_terminals = num_terminals)
    highway_costs = exctract_highway_costs(cost_matrix, num_terminals)
    partial_1 = 0
    partial_2 = 0
    for i in nodes_indices:
        for j in nodes_indices:
            for k in trailer_indices:
                for t in terminals_indices:
                    partial_1 += cost_matrix[i][j]*x[i][j][t][k]
    for t in terminals_indices[1:]:
        partial_2 += highway_costs[t]*w[t]

    return partial_1 + partial_2





