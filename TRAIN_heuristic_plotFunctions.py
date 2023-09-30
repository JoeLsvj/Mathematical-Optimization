# Lib
import random
import numpy as np
import gurobipy as gb
import matplotlib.pyplot as plt
import random
import networkx as nx 
import math
from itertools import chain

# Plot optimal solution
def plot_solution(nodes_matrix, num_terminals, num_trailers, w, x):

    # Indices
    num_nodes = len(nodes_matrix)
    nodes_indices = [i for i in range(num_nodes)]

    # Store (x,y) for each node
    x_coords = [point[0] for point in nodes_matrix]
    y_coords = [point[1] for point in nodes_matrix]

    colors = ['red'] * 1 + ['green'] * 1 + ['lime'] * (num_terminals - 1) + ['royalblue'] * (len(nodes_matrix) - (num_terminals + 1))
    plt.figure(figsize=(8,8))
    plt.scatter(x_coords, y_coords, color=colors)
    plt.title('Nodes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # Draw the routes of the road trains (between original terminal and secondary terminals), 
    # if they exist (green dashed lines)
    for t in nodes_indices[1 : num_terminals + 1 ]:
        if (w[t] != 0): plt.plot((x_coords[1],x_coords[t]), (y_coords[1],y_coords[t]), 'g--')
    # Draw the regular trucks routes (between terminals and customers), distinguishing between LT orders 
    # and LTL orders (with high or low pattern) (blue/light blue continuous lines)
    for k in range(num_trailers):
        k += 1
        for t in nodes_indices[1 : num_terminals + 1 ]:
            for i in nodes_indices[1:]:
                for j in nodes_indices[1:]:
                        #if (i < j):
                        # blue color for LT orders
                        if(x[i,j,t,k] == 2): plt.plot((x_coords[i],x_coords[j]), (y_coords[i],y_coords[j]), 'b-')
                        # light blue color for LTL orders
                        if(x[i,j,t,k] == 1): plt.plot((x_coords[i],x_coords[j]), (y_coords[i],y_coords[j]), 'c-')
                        
    legend_labels =     ['Road-train routes', 'TL orders', 'LTL orders']
    legend_handles =    [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color, markersize=8)
                        for label_color in ['g', 'b', 'c']]
    plt.legend(legend_handles, legend_labels, loc = 'best')
    #filename = 'GI_LT{TL}_LTL{LTL}_{pattern}_T{T}'.format(TL=num_costumers-LTL_orders, 
    #                                                    LTL=LTL_orders, pattern=LTL_pattern, T=num_terminals)
    #plt.savefig(filename+'.png')
    plt.show()
