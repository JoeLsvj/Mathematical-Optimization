# Libraries
import numpy as np
import random

import TRAIN_heuristic_objectiveFunction
import TRAIN_heuristic_localSearchOperators
import TRAIN_heuristic_manageData

# costumer -> get the associated terminal and trailer
def get_associated_terminal_and_trailer(costumer, z, terminals_indices, trailer_indices):
    for t in terminals_indices:
        for k in trailer_indices:
            if z[t][k][costumer] == 1:
                return t, k

# get the edges (undirected) of a LTL route, for a given terminal and a given trailer:        
def get_LTL_edges(x, terminal, trailer):
    route = []
    for i in range(len(x[:,0,0,0])):
        for j in range(len(x[0,:,0,0])):
            if i < j:
                if x[i,j,terminal,trailer] == 1:
                    route.append([i,j])
                elif x[i,j,terminal,trailer] == 2:
                    return None
    return route
    
# get the edges (directed) of a route, for a given terminal and a given trailer:
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


# get the edges (undirected) of a route, for a given terminal and a given trailer:
def get_route_edges_unordered(x, terminal, trailer):
    route = []
    for i in range(len(x[:,0,0,0])):
        for j in range(len(x[0,:,0,0])):
            if i < j:
                if x[i,j,terminal,trailer] == 1:
                    route.append([i,j])
                elif x[i,j,terminal,trailer] == 2:
                    # since the value of x is 2, I append twice for better representation
                    route.append([i,j])
                    route.append([i,j])
    return route
    

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


# Route, element_to_add -> a route with that node, with directed edges
def add_node_to_route_ordered(list_of_lists, element_to_add):
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


# Route, element_to_add -> a route with that node, with undirected edges
def add_node_to_route(route, element_to_add):
    if route != None:
        choosen_pair = random.choice(route)
        # filter the pairs directly here
        if (choosen_pair[0] > element_to_add):
            new_element_1 = [element_to_add, choosen_pair[0]]
        else: 
            new_element_1 = [choosen_pair[0], element_to_add]
        if (choosen_pair[1] > element_to_add):
            new_element_2 = [element_to_add, choosen_pair[1]]
        else: 
            new_element_2 = [choosen_pair[1], element_to_add]
        # check if the route is traversed twice (the edge appears once for the structure of 
        # the x matrix that has to be respected):
        if len(route) != 1:
            route.remove(choosen_pair)
        route.append(new_element_1)
        route.append(new_element_2)
        return route
    return []

# whole perturbation function:
def perturb(num_nodes, num_terminals, num_costumers, num_trailers, nodes_matrix, x, w, z, y, 
            costumers_indices, terminals_indices, trailer_indices, LTL_orders):
    R = []
    for t in terminals_indices:
        for k in trailer_indices:
            if (y[t, k] == 1):
                route = get_LTL_edges(x=x, terminal=t, trailer=k)
                if route != None:     # decomment for LTL routes edges
                    R.append(route)

    clients_in_routes = []
    for i in range(len(R)):
        clients_in_routes.append(tuple(get_client_numbers(R[i])))

    avg_demand = []
    for i in range(len(clients_in_routes)):
        avg = 0
        if (len(clients_in_routes[i])>0 and clients_in_routes[i] != None):
            for j in clients_in_routes[i]:
                avg += nodes_matrix[j,2]
            avg_demand.append(avg/len(clients_in_routes[i]))
        else: avg_demand.append(avg)

    clients_dict = dict(zip(clients_in_routes, avg_demand))
    clients_dict = sorted(clients_dict.items(), key=lambda x: x[1], reverse=True)
    first_route_clients = clients_dict[0][0]
    # get the route parameters associated with the first route (we can choose the first client of the route to get t and k)
    t1, k1 = get_associated_terminal_and_trailer(first_route_clients[0], z, terminals_indices, trailer_indices)
    #clients_dict.pop(0)

    cost_matrix = TRAIN_heuristic_objectiveFunction.compute_cost_matrix(nodes_matrix, 1, 1, num_terminals)
    min_distances = []
    selected_clients = []
    for i in first_route_clients:
        raw_distances = cost_matrix[i][:]
        # remove the distances between the nodes of the first route
        clean_distances = np.delete(raw_distances, first_route_clients)
        # decomment for LTL edges only:
        TL_clients = list(range(4,num_costumers-LTL_orders+4))
        clean_distances = np.delete(clean_distances, TL_clients)
        # remove the terminals positions
        clean_distances = clean_distances[4:]
        # append the correct minimum distance
        min_distances.append(np.min(clean_distances))
        # append the correct client/index 
        selected_clients.append(np.where(cost_matrix[i][:]==np.min(clean_distances))[0][0])
    selected_client = selected_clients[min_distances.index(min(min_distances))]

    for i in range(len(clients_in_routes)):
        if selected_client in clients_in_routes[i]:
            second_route_clients = clients_in_routes[i]
        else: return y, z, x, w
    # get the route parameters associated with the closest route (we can choose the first client of the route to get t and k)
    t2, k2 = get_associated_terminal_and_trailer(second_route_clients[0], z, terminals_indices, trailer_indices)

    # remove the selected routes:
    for u in range(len(x[:,0,0,0])):        # Clean x matrix: remove edges
        for v in range(len(x[0,:,0,0])):
            x[u,v,t1,k1] = 0
            x[u,v,t2,k2] = 0
    for u in costumers_indices:             # Clean z matrix
        z[t1,k1,u] = 0
        z[t2,k2,u] = 0
    # eventually, keep these void routes for the next step, do not delete them 
    y[t1,k1] = 0
    y[t2,k2] = 0

    # create the set V'
    V_prime = list(first_route_clients + second_route_clients)
    # recompute the set of routes with the applied changes:
    R = []
    for t in terminals_indices:
        for k in trailer_indices:
            if (y[t, k] == 1):
                route = get_LTL_edges(x=x, terminal=t, trailer=k)
                if route != None:
                    R.append(route)

    # find the list of available trailers:
    available_trailers = [trailer for trailer in trailer_indices]        # attention to the pointers makes in python
    #print(available_trailers)
    for t in terminals_indices:
        for k in trailer_indices:
            if (y[t,k] == 1):
                available_trailers.remove(k)
    # list of void routes 
    void_routes = [(t1, k1), (t2, k2)]

    while len(V_prime) > 0:
        # choose a random route from R
        random_route = random.choice(R)
        # choose a random customer from V'
        random_costumer = random.choice(V_prime)
        # remove the costumer from the set V':
        V_prime.remove(random_costumer)
        random_edges = get_client_numbers(random_route)
        if len(random_route) > 0:   # check for empty list case
            t_random, k_random = get_associated_terminal_and_trailer(random_edges[0], z, terminals_indices, trailer_indices)
        else: 
            t_random, k_random = random.choice(void_routes)
            void_routes.remove((t_random, k_random))
        # try to relocate
        if ( sum([nodes_matrix[j][2] for j in random_edges] + [nodes_matrix[random_costumer][2]]) <= 24 ):
            # feasibility
            if len(random_route) > 0:   # check for the empty list
                new_route = add_node_to_route(random_route, random_costumer)
            else: new_route = [[t_random, random_costumer], [t_random, random_costumer]]
            # update the set of all routes:
            R.remove(random_route)
            R.append(new_route)
            # update all the decision varibales:
            for u in range(len(x[:,0,0,0])): # Clean x matrix
                for v in range(len(x[0,:,0,0])):
                    x[u,v,t_random,k_random] = 0  
            for pair in new_route: # add new route edges
                x[pair[0], pair[1],t_random,k_random] += 1
            z[t_random][k_random][random_costumer] = 1
            # check for correctness:
            #print(R)
            #R_test = []
            #for t in terminals_indices:
            #    for k in trailer_indices:
            #        route = get_LTL_edges(x=x, terminal=t, trailer=k)
            #        if route != None:
            #            R_test.append(route)
            #print(R_test)

        else:
            # infeasibility: start from the main
            k_random = random.choice(available_trailers)
            # update the decision variables
            x[1,random_costumer,1,k_random] = 2 # here the edge is traversed twice (for the construction of the x matrix)
            #x[1,random_costumer,1,k_random] = 1
            z[1, k_random, random_costumer] = 1
            y[1, k_random] = 1
            # update the set of all routes:
            new_route = [[1, random_costumer], [1, random_costumer]]
            #new_route = [[1, random_costumer],[random_costumer, 1]]
            R.append(new_route)
    
    return y, z, x, w
