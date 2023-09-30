# Libs
import random
import numpy as np
import random
import math
import networkx as nx
import re
import copy

# Our libs
import TRAIN_heuristic_plotFunctions
import TRAIN_heuristic_randomSolution
import TRAIN_heuristic_manageData
import TRAIN_heuristic_objectiveFunction
import TRAIN_heuristic_localSearchOperators



# INP: ...
# OUT: two list of LTL routes
def find_two_LTL_routes(costumers_indices, x, z, terminals_indices,trailer_indices):
    list_of_routes = list()
    J_temp = list(costumers_indices)
    num_ltl_routes_found = 0
    while num_ltl_routes_found < 2 and J_temp != []:
        j = random.choice(J_temp)

        t,k = TRAIN_heuristic_localSearchOperators.get_associated_terminal_and_trailer(costumer=j, z = z, terminals_indices=terminals_indices, trailer_indices=trailer_indices)
        num_elements_in_j_route = np.sum(x[:,:,t,k])

        if num_elements_in_j_route >= 3:
            list_of_routes.append(    TRAIN_heuristic_localSearchOperators.get_clients_in_route(terminal = t, trailer = k, costumers_indices = costumers_indices, z = z)   )
            for element in list_of_routes[num_ltl_routes_found]:
                J_temp.remove(element)
            num_ltl_routes_found += 1
        else:
            J_temp.remove(j)

    if len(list_of_routes) < 2:
        return False, False
    return list_of_routes[0], list_of_routes[1]


# Parameters
def remove_edges_of_two_LTL_routes(route1,route2,x,z, terminals_indices, trailer_indices):
    
    # code
    x_temp = copy.deepcopy(x)
    z_temp = copy.deepcopy(z)
    t_1,k_1 = TRAIN_heuristic_localSearchOperators.get_associated_terminal_and_trailer(costumer=route1[0], z = z_temp, terminals_indices=terminals_indices, trailer_indices=trailer_indices)
    t_2,k_2 = TRAIN_heuristic_localSearchOperators.get_associated_terminal_and_trailer(costumer=route2[0], z = z_temp, terminals_indices=terminals_indices, trailer_indices=trailer_indices)

    edges_of_route1 = TRAIN_heuristic_localSearchOperators.get_route_edges(x = x_temp, terminal=t_1, trailer=k_1)
    edges_of_route2 = TRAIN_heuristic_localSearchOperators.get_route_edges(x = x_temp, terminal=t_2, trailer=k_2)

    for couple in edges_of_route1:
        x_temp[couple[0],couple[1],t_1,k_1] = 0
        z_temp[t_1,k_1,couple[0]] = 0
        z_temp[t_1,k_1,couple[1]] = 0

    for couple in edges_of_route2:
        x_temp[couple[0],couple[1],t_2,k_2] = 0
        z_temp[t_2,k_2,couple[0]] = 0
        z_temp[t_2,k_2,couple[1]] = 0

    return x_temp,z_temp,t_1,k_1,t_2,k_2



def build_groups_with_numbers_less_then_a_threshold(client_and_demand, Q):
    group_a = []
    group_b = []
    sum_a = 0
    sum_b = 0

    random.shuffle(client_and_demand)
    
    for num in client_and_demand:
        # Try to add the number to group A if it doesn't exceed the threshold
        if sum_a + num[1] <= Q:
            group_a.append([num[0],num[1]])
            sum_a += num[1]
        else:
            # If adding the number to group A exceeds the threshold, add it to group B
            group_b.append([num[0],num[1]])
            sum_b += num[1]
    
    return group_a, group_b



# Build new routes
def build_new_routes(route1,route2, t_1,k_1,t_2,k_2, nodes_matrix, Q, x, z):

    clients_not_served = route1 + route2
    clients_and_demands = []
    group_a_is_fill = True
    group_b_is_fill = True
    
    for client in clients_not_served:
        clients_and_demands.append([client, nodes_matrix[client][2]])

    group_a, group_b = build_groups_with_numbers_less_then_a_threshold(client_and_demand=clients_and_demands, Q = Q)
    
    j_first = t_1
    while group_a_is_fill:
        pair = random.choice(group_a)
        group_a.remove(pair)
        j_second = pair[0]
        z[t_1, k_1, j_second] = 1

        if j_first < j_second:
            x[j_first,j_second,t_1,k_1] = 1
        else:
            x[j_second,j_first, t_1,k_1] = 1

        if group_a == []:
            x[t_1, j_second, t_1, k_1] = 1
            group_a_is_fill = False
        else:
            j_first = j_second

    j_first = t_2
    while group_b_is_fill:
        pair = random.choice(group_b)
        group_b.remove(pair)
        j_second = pair[0]
        z[t_2, k_2, j_second] = 1

        if j_first < j_second:
            x[j_first,j_second,t_2,k_2] = 1
        else:
            x[j_second,j_first, t_2,k_2] = 1

        if group_b == []:
            x[t_2, j_second, t_2, k_2] = 1
            group_b_is_fill = False
        else:
            j_first = j_second
    
    return x,z



# Find two LTL routes randomly, remove all the edges, re build the edges in a random way that respect the capacity constraint, if this is better store it 
def SL_LTL_change_lc(
        num_nodes, num_terminals, num_costumers, num_trailers, nodes_matrix,
        c1, c2, x, w, z, y, costumers_indices, terminals_indices, trailer_indices, r, Q
        ):
    
    # Compute the current obj function
    best_f = TRAIN_heuristic_objectiveFunction.objective_function( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers,
                                                          num_trailers = num_trailers, nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x, w = w )
    
    
    # Find two routes randomly
    route1, route2 = find_two_LTL_routes(costumers_indices=costumers_indices, x= x, z=z,terminals_indices=terminals_indices,trailer_indices=trailer_indices)

    # Not exist more then 1 LTL route
    if route1 == False:
        return y,z,x,w

    # 5 trials to improve
    for _ in range(30):

        # Remuve all edges of this 2 reoute
        x_temp,z_temp, t_1,k_1,t_2,k_2 = remove_edges_of_two_LTL_routes(route1=route1,route2=route2, x = x, z = z, terminals_indices=terminals_indices,trailer_indices=trailer_indices)
        # Build new edges of the two routes (respecting the capacity constraint)
        x_temp, z_temp = build_new_routes(route1 = route1, route2 = route2, t_1 = t_1,k_1 = k_1,t_2 = t_2,k_2 = k_2, nodes_matrix = nodes_matrix, Q = Q, x = x_temp, z = z_temp)   
        # Compute the current obj function
        new_f = TRAIN_heuristic_objectiveFunction.objective_function( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, 
                                                                    nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x_temp, w = w )
        
        # If the current obj function is better then the previous, then store the current configuration and end the function
        if new_f < best_f:
            best_f = new_f
            x = copy.deepcopy(x_temp)
            z = copy.deepcopy(z_temp)
            return y,z,x,w

    return y,z,x,w

