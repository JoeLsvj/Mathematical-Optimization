# INDEX
# Exchange
# Relocate
# LOCAL SEARCH functions


# Libraries
import numpy as np
import random
import copy
import TRAIN_heuristic_objectiveFunction
import TRAIN_heuristic_LS_LTL_SL
#import TRAIN_heuristic_manageData
#import TRAIN_heuristic_plotFunctions



# EXCHANGE

# costumer -> associated terminal and trailer
def get_associated_terminal_and_trailer(costumer, z, terminals_indices, trailer_indices):
    for t in terminals_indices:
        for k in trailer_indices:
            if z[t][k][costumer] == 1:
                return t, k
            
def get_route_edges(x, terminal, trailer):
    route = []
    for i in range(len(x[:,0,0,0])):
        for j in range(len(x[0,:,0,0])):
            if i < j:
                if x[i,j,terminal,trailer] == 1:
                    route.append([i,j])
    return route
    
def get_route_edges_ordered(x, terminal, trailer):
    route = []
    for i in range(len(x[:,0,0,0])):
        for j in range(len(x[0,:,0,0])):
            if x[i,j,terminal,trailer] == 1:
                route.append([i,j])
            elif x[i,j,terminal,trailer] == 2:
                route.append([i,j])
                route.append([j,i])
	# order here directly the edges of the path
    if (len(route)) > 0:
        edges = get_unique_numbers_as_list(route)
        ordered_route = [route[0]]
        p = 0
        edge = route[0][1]
        while p < len(route)-1:
            for k in range(1, len(route)):
                for l in range(2):
                    if route[k][l] == edge:
                        ordered_route.append(route[k])
                        edges.remove(edge)
                        edge = list(set(route[k]) & set(edges))[0]
                        route[k] = [0,0]
                        p += 1
        return order_path(ordered_route)
    else: return route

# List_of_lists -> list of numbers of the list_of_lists (list of edges / path -> list of visited nodes)
def get_unique_numbers_as_list(list_of_lists):
    flat_list = [number for sublist in list_of_lists for number in sublist]
    unique_numbers = list(set(flat_list))
    return unique_numbers
    
# List_of_lists -> list of numbers of the list_of_lists (list of edges / path -> list of visited nodes)
def get_client_numbers(route):
    flat_list = [number for sublist in route for number in sublist if number >= 4]
    client_numbers = list(set(flat_list))
    return client_numbers

## [[1,2][3,1]] -> [[1,2][1,3]] (i < j fo all pair)
def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if pair[0] < pair[1]:
            filtered_pairs.append(pair)
        elif pair[1] < pair[0]:
            filtered_pairs.append([pair[1],pair[0]])
    return filtered_pairs

# array_of_array, key -> choose a random_element c, swap key with c in all the array_of_array
def swap_elements(edges_list, costumer_1, costumer_2 = None):
    swapped_arrays = []
    if costumer_2 == None:
        for _ in range(5): # Try at most 5 time to take an element different from the key
            random_pair = random.choice(edges_list)
            costumer_2 = random.choice(random_pair)
            # remember that the costumers nodes have indices >= 4 
            # (exclude to exchange the terminal node position since it is always the starting point)
            if costumer_2 != costumer_1 and costumer_2 >= 4:
                break
    for arr in edges_list:
        swapped_arr = [costumer_2 if element == costumer_1 else (costumer_1 if element == costumer_2 else element) for element in arr]
        swapped_arrays.append(swapped_arr)
    swapped_arrays_corrected = filter_pairs(swapped_arrays)
    return swapped_arrays_corrected

# INP: ...
# OUT: two list of LTL routes
def find_two_LTL_routes(costumers_indices, x, z, terminals_indices,trailer_indices):
    list_of_routes = list()
    J_temp = list(costumers_indices)
    num_ltl_routes_found = 0
    while num_ltl_routes_found < 2 and J_temp != []:
        j = random.choice(J_temp)

        t,k = get_associated_terminal_and_trailer(costumer=j, z = z, terminals_indices=terminals_indices, trailer_indices=trailer_indices)
        num_elements_in_j_route = np.sum(x[:,:,t,k])

        if num_elements_in_j_route >= 3:
            list_of_routes.append(    get_clients_in_route(terminal = t, trailer = k, costumers_indices = costumers_indices, z = z)   )
            for element in list_of_routes[num_ltl_routes_found]:
                J_temp.remove(element)
            num_ltl_routes_found += 1
        else:
            J_temp.remove(j)

    if len(list_of_routes) == 0:
        return False, False
    if len(list_of_routes) == 1:
        return list_of_routes[0], False
    return list_of_routes[0], list_of_routes[1]


# exchange operator function
def exchange(num_nodes, num_terminals, num_costumers, num_trailers, nodes_matrix,
                c1, c2, x, w, z, y, costumers_indices, terminals_indices, trailer_indices, r, Q):

    current_f = TRAIN_heuristic_objectiveFunction.objective_function(
                    num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, nodes_matrix = nodes_matrix,
                    c1 = c1, c2 = c2, x = x, w = w
                    )

    route1, route2 = find_two_LTL_routes(costumers_indices=costumers_indices, x= x, z=z,terminals_indices=terminals_indices,trailer_indices=trailer_indices)

    if route2 == False: 
        return y,z,x,w
    if route1 == False:
        list_of_routes = [route1]
    if route1!= False and route2 != False:
        list_of_routes = [route1,route2]

    for route in list_of_routes:
        j = random.choice(route)
        # Asccoiated terminal and trailer
        t,k = get_associated_terminal_and_trailer(costumer = j, z = z, terminals_indices = terminals_indices, trailer_indices = trailer_indices)
        # Number of edegs in the route
        route_edges_number = np.sum(x[:,:,t,k])
        # Swap depending on the number of edegs
        if route_edges_number > 3: # if num_edegs = 2 or 3, do nothing  
            route = get_route_edges(x = x, terminal = t, trailer = k)
            numbers = get_unique_numbers_as_list(list_of_lists = route)

            x_temp = np.copy(x)
            for a in numbers:
                swapped_route = swap_elements(edges_list = route, costumer_1 = j, costumer_2 = a)
                for u in numbers: # Clean x matrix
                    for v in numbers:
                        x_temp[u,v,t,k] = 0  
                for pair in swapped_route: # add new route edges
                    x_temp[pair[0], pair[1],t,k] += 1

                f = TRAIN_heuristic_objectiveFunction.objective_function(
                    num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, nodes_matrix = nodes_matrix,
                    c1 = c1, c2 = c2, x = x_temp, w = w
                    )

                if f < current_f:
                    current_f = f
                    x = np.copy(x_temp)

    return y,z,x,w



# RELOCATE

# Terminal t, trailer k -> all the costumer in the route of k from t
# the costumers are ordered according to the index, not the real path followed
def get_clients_in_route(terminal, trailer, costumers_indices, z):
    clients_in_route = []
    for j in costumers_indices:
        if z[terminal][trailer][j] == 1:
            clients_in_route.append(j)
    return clients_in_route
    
# Route, element_to_remove -> a route with that node
def add_node_to_route(list_of_lists, element_to_add):
    if list_of_lists != None:
        new_list_of_lists = order_path(list_of_lists)
        choosen_pair = random.choice(new_list_of_lists)
        choosen_pair_index = new_list_of_lists.index(choosen_pair)
        new_element_1 = [choosen_pair[0], element_to_add]
        new_element_2 = [element_to_add, choosen_pair[1]]
        new_list_of_lists.remove(choosen_pair)
        new_list_of_lists.insert(choosen_pair_index, new_element_1)
        new_list_of_lists.insert(choosen_pair_index+1, new_element_2)
        return new_list_of_lists
    return []

# Route, element_to_remove -> a route without that node
def remove_node_from_route(list_of_lists, element_to_remove):
    new_list_of_lists = []
    new_element = [0,0]
    counter = 0
    if list_of_lists != None:
        for pair in list_of_lists:
            if (pair[0] != element_to_remove) and (pair[1] != element_to_remove):
                new_list_of_lists.append( [pair[0], pair[1]] )
            elif pair[0] == element_to_remove:
                new_element[counter] = pair[1]
                counter += 1
            elif pair[1] == element_to_remove:
                new_element[counter] = pair[0]
                counter += 1
        if new_element[0] > new_element[1]:
            temp = new_element[0]
            new_element[0] = new_element[1]
            new_element[1] = temp
        new_list_of_lists.append(new_element)
        if len(new_list_of_lists) == 1:
            new_list_of_lists.append(new_element)
        return new_list_of_lists
    else:
        return []

# unordered path of nodes like [[1, 2], [0, 2], [0, 3], [1, 3]] 
# -> the orderd path with the correct verse of edges [[1, 2], [2, 0], [0, 3], [3, 1]]
def order_path(path):
    if path != None:
        ordered_path = list(path)
        if (ordered_path[0][0] > ordered_path[0][1]): 
            temp = ordered_path[0][1]
            ordered_path[0][1] = ordered_path[0][0]
            ordered_path[0][0] = temp
        for i in range(1, len(ordered_path)):
            if (ordered_path[i][0] != ordered_path[i-1][1]):
                temp = ordered_path[i][1]
                ordered_path[i][1] = ordered_path[i][0]
                ordered_path[i][0] = temp
        return ordered_path
    else:
        return []


def build_available_places_array(num_terminals):
    return [0 for _ in range(num_terminals + 1)]


"""
def build_available_places_array(num_terminals, y, terminals_indices, trailer_indices):
    available_places = [0 for _ in range(num_terminals + 1)]
    for t in [2,3]:
        trailer_count = 0
        for k in trailer_indices:
            if y[t, k] == 1:
                trailer_count += 1
        available_places[t] = trailer_count % 2
    return available_places
"""

def check_TL_route_change_chance(w,r, new_assigned_terminal_id, available_places):
    if new_assigned_terminal_id == 1 or (np.sum(w) < r) or (available_places[new_assigned_terminal_id] == 1):
        return True
    else:
        return False
    
def remove_TL_from_route(past_assigned_terminal_id, j, k, available_places, w,x,y,z):
    # about truck
    x[past_assigned_terminal_id,j,past_assigned_terminal_id, k] = 0
    y[past_assigned_terminal_id,k] = 0
    z[past_assigned_terminal_id,k,j] = 0
    # about road-train
    if past_assigned_terminal_id != 1:
        if available_places[past_assigned_terminal_id] == 0:
            available_places[past_assigned_terminal_id] = 1
        elif available_places[past_assigned_terminal_id] == 1:
            w[past_assigned_terminal_id] -= 1
            available_places[past_assigned_terminal_id] = 0
    return x, available_places, w, y, z

def add_TL_route(x, new_assigned_terminal_id,j,k,available_places,w,y,z):
    x[new_assigned_terminal_id,j,new_assigned_terminal_id,k] = 2
    y[new_assigned_terminal_id,k] = 1
    z[new_assigned_terminal_id,k,j] = 1
    if new_assigned_terminal_id != 1: # here i put 0 (it is or 0 or 1 idk)
        if available_places[new_assigned_terminal_id] == 1:
            available_places[new_assigned_terminal_id] = 0
        elif available_places[new_assigned_terminal_id] == 0:
            w[new_assigned_terminal_id] += 1
            available_places[new_assigned_terminal_id] = 1
    return x, available_places, w, y, z


def TL_route_change_chance(costumers_indices, z, terminals_indices, trailer_indices, x, w, r, available_places, y):

    # Find a TL costumer
    while True:
        #trials += 1
        j = random.choice(costumers_indices)
        t,k = get_associated_terminal_and_trailer(costumer = j, z = z, terminals_indices = terminals_indices, trailer_indices = trailer_indices)
        route_edges_number = np.sum(x[:,:,t,k])
        if route_edges_number == 2:
            break

    terminals_indices_temp = list(terminals_indices)
    terminals_indices_temp.remove(t)
    t_new = random.choice(terminals_indices_temp)
    # Check if it is possible
    if check_TL_route_change_chance(w = w, r = r, new_assigned_terminal_id = t_new, available_places = available_places) == True:
        # remove
        x, available_places, w, y, z = remove_TL_from_route(past_assigned_terminal_id = t, j = j, k = k, available_places = available_places, w = w,x = x, y = y, z = z)
        # add
        x, available_places,w, y, z = add_TL_route(x = x, new_assigned_terminal_id = t_new ,j = j,k = k,available_places = available_places,w = w, y = y, z = z)
        return x, available_places, w, y, z
    
    return x, available_places, w, y, z


def correct_odd_highway_paths(
        num_nodes, num_terminals, num_costumers, num_trailers, nodes_matrix,
        c1, c2, x, w, z, y, costumers_indices, terminals_indices, trailer_indices, available_places
        ):
    # Correct situation like: a road train is sent with only one trailer
    indices_of_nodes = [i for i in range(num_nodes)] 
    J = indices_of_nodes[num_terminals + 1:] # costumers indices
    K = [i for i in range(num_trailers + 1)] # trailers indices

    for t in [2,3]:
        costumer_removed = []
        if available_places[t] == 1:
            # Remove the road train to t
            w[t] -= 1
            available_places[t] = 0
            # take a trailer k assigned to t
            for k in K:
                if y[t][k] == 1:
                    assigned_trailer = k
                    break
            # reset all the route of k from t
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

    return y,z,x,w,available_places


# relocate operator function        
def TL_relocate(num_nodes, num_terminals, num_costumers, num_trailers, nodes_matrix,
    c1, c2, x, w, z, y, costumers_indices, terminals_indices, trailer_indices,r, Q
    ):

    current_f = TRAIN_heuristic_objectiveFunction.objective_function( 
        num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, nodes_matrix = nodes_matrix,
        c1 = c1, c2 = c2, x = x, w = w
        )
    
    toy_costumers_indeces = list(costumers_indices)
    #available_places = build_available_places_array(num_terminals = num_terminals, y=y, terminals_indices=terminals_indices, trailer_indices=trailer_indices)
    available_places = build_available_places_array(num_terminals = num_terminals)
    iterations = 0

    while toy_costumers_indeces != []: # Repeat for all costuemers
        # Instad of sample randomly, we can take the j depenidng on the position in the dataset
        j = random.choice(toy_costumers_indeces) # choose a costumer
        toy_costumers_indeces.remove(j)
        t,k = get_associated_terminal_and_trailer(costumer = j, z = z, terminals_indices = terminals_indices, trailer_indices = trailer_indices)
        route_edges_number = np.sum(x[:,:,t,k])
        
        if route_edges_number == 2:
            #iterations += 1
            x_temp = np.copy(x)
            y_temp = np.copy(y)
            z_temp = np.copy(z)
            w_temp = np.copy(w)
            available_places_temp = np.copy(available_places)

            for _ in range(5):
                x_temp, available_places_temp, w_temp, y_temp, z_temp = TL_route_change_chance(
                    costumers_indices = costumers_indices,
                    z = z_temp, 
                    terminals_indices = terminals_indices,
                    trailer_indices = trailer_indices,
                    x = x_temp,
                    w = w_temp,
                    r = r,
                    available_places = available_places_temp,
                    y = y_temp
                )

            f = TRAIN_heuristic_objectiveFunction.objective_function( 
                        num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, nodes_matrix = nodes_matrix,
                        c1 = c1, c2 = c2, x = x_temp, w = w_temp
                        )

            y_temp,z_temp,x_temp,w_temp, available_places_temp = correct_odd_highway_paths(
                        num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, nodes_matrix = nodes_matrix,
                        c1 = c1, c2 = c2, x = x_temp, w = w_temp, z = z_temp, y = y_temp, costumers_indices = costumers_indices,
                        terminals_indices = terminals_indices, trailer_indices = trailer_indices, available_places = available_places_temp
                        )
            
            if f < current_f:
                current_f = f
                x = np.copy(x_temp)
                y = np.copy(y_temp)
                z = np.copy(z_temp)
                w = np.copy(w_temp)
                available_places = np.copy(available_places_temp)
        
    return y,z,x,w
    

    
# Actual LOCAL SEARCH complete functions:

def multi_iterated_local_search( num_nodes,num_terminals,num_costumers,num_trailers,nodes_matrix,c1,c2,
                                x,w,z,y,costumers_indices,terminals_indices,trailer_indices,r, Q, iteration_number = 10
                                ):
    
    f = TRAIN_heuristic_objectiveFunction.objective_function( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers,
                                                              num_trailers = num_trailers, nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x, w = w )
    you_can_go_out = False

    while iteration_number > 0:
        iteration_number -= 1
        list_of_localSearch = list([TL_relocate, exchange, TRAIN_heuristic_LS_LTL_SL.SL_LTL_change_lc])

        while you_can_go_out == False:
            
            ls = random.choice(list_of_localSearch)
    
            y_temp,z_temp,x_temp,w_temp = ls( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers,
                                             nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x, w = w, z = z, y = y, costumers_indices = costumers_indices,
                                             terminals_indices = terminals_indices, trailer_indices = trailer_indices, r = r, Q = Q)

            new_f = TRAIN_heuristic_objectiveFunction.objective_function( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, 
                                                                          nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x_temp, w = w_temp )

            if new_f < f:
                f = new_f
                x = np.copy(x_temp)
                y = np.copy(y_temp)
                z = np.copy(z_temp)
                w = np.copy(w_temp)

                list_of_localSearch = list([TL_relocate, exchange, TRAIN_heuristic_LS_LTL_SL.SL_LTL_change_lc])

            else:
                list_of_localSearch.remove(ls)
                if len(list_of_localSearch) == 0: you_can_go_out = True
    
    return y,z,x,w
    
    

    
def local_search( num_nodes,num_terminals,num_costumers,num_trailers,nodes_matrix,c1,c2,
                  x,w,z,y,costumers_indices,terminals_indices,trailer_indices,r, Q, iteration_number = 10
                  ):
    
    f = TRAIN_heuristic_objectiveFunction.objective_function( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers,
                                                              num_trailers = num_trailers, nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x, w = w )
    
    while iteration_number > 0:
        iteration_number -= 1
        l = random.choice([TL_relocate, exchange, TRAIN_heuristic_LS_LTL_SL.SL_LTL_change_lc])
  
        y_temp,z_temp,x_temp,w_temp = l( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers,
                                         nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x, w = w, z = z, y = y, costumers_indices = costumers_indices,
                                         terminals_indices = terminals_indices, trailer_indices = trailer_indices, r = r, Q = Q)
        
        new_f = TRAIN_heuristic_objectiveFunction.objective_function( num_nodes = num_nodes, num_terminals = num_terminals, num_costumers = num_costumers, num_trailers = num_trailers, 
                                                                      nodes_matrix = nodes_matrix, c1 = c1, c2 = c2, x = x_temp, w = w_temp )
        
        if new_f < f:
            f = new_f
            x = np.copy(x_temp)
            y = np.copy(y_temp)
            z = np.copy(z_temp)
            w = np.copy(w_temp)
    
    return y,z,x,w


