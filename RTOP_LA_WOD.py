import numpy as np
import gurobipy as gb
import matplotlib.pyplot as plt
import random
import networkx as nx 
import math
from itertools import chain
import re

class RTOP_LA(object):

    def __init__(self, nodes, num_costumers, num_terminals, LTL_orders, LTL_pattern, costumer_demands, number_of_trailers, trailer_capacity, number_experienced_drivers, c1, c2):
        self.train_model = gb.Model()
        self.train_model.setParam('TimeLimit', 15*60)
        self.train_model.modelSense = gb.GRB.MINIMIZE

        self.nodes = nodes
        self.num_costumers = num_costumers
        self.num_terminals = num_terminals
        self.LTL_orders = LTL_orders
        self.LTL_pattern = LTL_pattern
        self.q = costumer_demands
        self.number_of_trailers = number_of_trailers
        self.Q = trailer_capacity
        self.r = number_experienced_drivers
        self.c1 = c1
        self.c2 = c2

        cost_matrix = self.compute_cost_matrix()
        highway_costs = self.exctract_highway_costs(cost_matrix)
        self.c = cost_matrix
        self.c_bar = highway_costs

        # Define the sets:
        self.J = range(self.num_terminals, len(self.nodes))
        # without dummy node: in this case V is the set of all the terminals and all the customers.
        # this set is required in order to close the routes over a selected terminal, and always on the dummy node
        # as it is done in the paper
        self.V = range(0, self.num_terminals+self.num_costumers)
        self.T = range(0, self.num_terminals)
        self.K = range(1, self.number_of_trailers+1)
    
    def setup(self):
        # Add variables:
        self.x = self.train_model.addVars(
            [(i, j, t, k) for k in self.K for t in self.T for i in self.V for j in self.J if (i < j)],
            vtype = gb.GRB.INTEGER,
            # specify that the values are in {0,1,2} with lower bound and upper bound
            lb = 0,
            ub = 2
        )
        self.w = self.train_model.addVars(
            [t for t in self.T[1:]],
            vtype = gb.GRB.INTEGER,
            lb = 0,
            ub = self.r
        )
        self.y = self.train_model.addVars(
            [(t,k) for t in self.T for k in self.K],
            vtype = gb.GRB.BINARY
        )
        self.z = self.train_model.addVars(
            [(t,k,j) for t in self.T for k in self.K for j in self.J], 
            vtype = gb.GRB.BINARY
        )

        # objective function:
        self.train_model.setObjective(
            gb.quicksum(self.c[i, j] * self.x[i, j, t, k] for k in self.K for t in self.T for i in self.V for j in self.J if(i < j)) +
            gb.quicksum(self.c_bar[t] * self.w[t] for t in self.T[1:])
        )

        # Add the constraints to the model.
        # Constraint 1: \sum_{t \in T \setminus \{1\}} w_t <= r
        self.train_model.addConstr(
            gb.quicksum(self.w[t] for t in self.T[1:]) <= self.r
        )
        # Constraint 2: \sum_{k \in K} y_{tk} = 2 w_t \hspace{1em} \forall t \in T \setminus \{1\}
        for t in self.T[1:]:
            self.train_model.addConstr(
                gb.quicksum(self.y[t, k] for k in self.K) == 2 * self.w[t] 
            )
        # Constraint 3: \sum_{t \in T} y_{tk} <= 1 \hspace{1em} \forall k\in K
        for k in self.K:
            self.train_model.addConstr(
                gb.quicksum(self.y[t, k] for t in self.T) <= 1
            )
        # Constraint 4: \sum_{j \in J} q_j z_{tkj} <= Q y_{tk} \hspace{1em} \forall k \in K \hspace{1em} \forall t \in T
        for k in self.K:
            for t in self.T:
                self.train_model.addConstr(
                    gb.quicksum(self.q[j] * self.z[t, k, j] for j in self.J) <= self.Q * self.y[t, k]
                )
        # Constraint 5: \sum_{t \in T} \sum_{k \in K} z_{tkj} = 1 \hspace{1em} \forall j \in J
        for j in self.J:
            self.train_model.addConstr(
                gb.quicksum(self.z[t, k, j] for t in self.T for k in self.K) == 1
            )
        # Constraint 6: \sum_{i \in V \text{ and } i<j} x_{ijtk} + \sum_{i \in V \text{ and } i>j } x_{jitk} = 2 z_{tkj} 
        #               \hspace{1em}\forall j  \in J \hspace{1em} \forall t \in T \hspace{1em} \forall k \in K
        for j in self.J:
            for t in self.T:
                for k in self.K:
                    self.train_model.addConstr(
                        gb.quicksum(self.x[i, j, t, k] for i in list(self.J)+[t] if i < j) +
                        gb.quicksum(self.x[j, i, t, k] for i in list(self.J)+[t] if i > j) == 2 * self.z[t, k, j]
                    )
        # Constraint 7: \sum_{j \in J} x_{0jtk} = 2y_{tk} \hspace{1em} \forall k \in K \hspace{1em} \forall t \in T
        for k in self.K:
            for t in self.T:
                self.train_model.addConstr(
                    gb.quicksum(self.x[t, j, t, k] for j in self.J) == 2 * self.y[t, k]
                )
        # Constraint 8:     \sum_{i \in S} \sum_{j \in S \text{ and } i < j} x_{ijkt} \leq |S| - 1 
        #                   \hspace{1em} \forall S \subseteq J \hspace{1em} \forall k \in K \hspace{1em} \forall t \in  T
        for S in self.powerset_without_emptyset():
            # constraint on the powerset of J:
            if (sum(self.q[j] for j in S) < self.Q):
                for t in self.T:
                    for k in self.K:
                        self.train_model.addConstr(
                            gb.quicksum(self.x[i, j, t, k] for i in S for j in S if(i < j)) <= len(S) - 1
                        )
    def optimize(self):
        self.train_model.optimize()

    # Plot optimal solution
    def plot_solution(self):
        x_coords = [point[0] for point in self.nodes]
        y_coords = [point[1] for point in self.nodes]
        colors =   ['green'] * 1 + ['lime'] * (self.num_terminals - 1) + ['royalblue'] * (len(self.nodes) - self.num_terminals)
        plt.figure(figsize=(8,8))
        plt.scatter(x_coords, y_coords, color=colors)
        plt.title('Nodes')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        # draw the routes of the road trains (between original terminal and secondary terminals), 
        # if they exist (green dashed lines)
        for t in self.T[1:]:
            if (self.w[t].x != 0): plt.plot((x_coords[0],x_coords[t]), (y_coords[0],y_coords[t]), 'g--')
        # draw the regular trucks routes (between terminals and customers), distinguishing between LT orders 
        # and LTL orders (with high or low pattern) (blue/light blue continuous lines)
        for k in self.K:
            for t in self.T:
                for j in self.J:
                    for i in self.V:
                        if (i < j):
                            # blue color for LT orders
                            if(self.x[i,j,t,k].x == 2): plt.plot((x_coords[i],x_coords[j]), (y_coords[i],y_coords[j]), 'b-')
                            # light blue color for LTL orders
                            if(self.x[i,j,t,k].x == 1): plt.plot((x_coords[i],x_coords[j]), (y_coords[i],y_coords[j]), 'c-')
        legend_labels =     ['road-train routes', 'LT orders', 'LTL orders']
        legend_handles =    [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_color, markersize=8)
                            for label_color in ['g', 'b', 'c']]
        plt.legend(legend_handles, legend_labels, loc = 'best')
        filename = 'GI_LT{TL}_LTL{LTL}_{pattern}_T{T}'.format(TL=self.num_costumers-self.LTL_orders, 
                                                            LTL=self.LTL_orders, pattern=self.LTL_pattern, T=self.num_terminals)
        plt.savefig(filename+'.png')
        plt.show()

    # save the instance/configuration of the problem (eventually to reproduce the results)
    def save_instance(self):
        filename = 'GI_LT{TL}_LTL{LTL}_{pattern}_T{T}'.format(TL=self.num_costumers-self.LTL_orders, LTL=self.LTL_orders, 
                                                                pattern=self.LTL_pattern, T=self.num_terminals)
        FILE = open(filename, 'w')
        FILE.write( '{r}  {Q}  {c1}  {c2}  {num_costumers}  {TL}  {LTL}  {pattern}  {T}'.format(
                    r=self.r, Q=self.Q, c1=self.c1, c2=self.c2, num_costumers=self.num_costumers, 
                    TL=self.num_costumers-self.LTL_orders, 
                    LTL=self.LTL_orders, pattern=self.LTL_pattern, T=self.num_terminals))
        FILE.write("\n\n")
        for point in self.nodes:
            FILE.write('{x}  {y}\n'.format(x=point[0], y=point[1]))
        FILE.write("\n")
        for j in range(self.num_terminals+1, len(self.nodes)):
            FILE.write('{q}  '.format(q=self.q[j]))
        FILE.close()

    def model_reset(self):
        self.train_model = gb.Model()

    # Powerset function without the empty set [] as element
    def powerset_without_emptyset(self):
        power_set = [[]]
        for element in self.J:
            new_subsets = [subset + [element] for subset in power_set]
            power_set.extend(new_subsets)
        power_set = [x for x in power_set if x != []]
        return power_set

    def compute_cost_matrix(self):
        num_nodes = len(self.nodes)
        cost_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                x1, y1 = self.nodes[i]
                x2, y2 = self.nodes[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Highway case
                if (i == 1 and j != 0 and j < self.num_terminals) or (j == 1 and i != 0 and i < self.num_terminals):
                    cost_matrix[i][j] = round(distance * self.c2, 2)
                # Standard road case
                else:
                    cost_matrix[i][j] = round(distance * self.c1, 2)
        return cost_matrix

    def exctract_highway_costs(self, cost_matrix):
        highway_cost = cost_matrix[0][0 : self.num_terminals]
        return highway_cost
        # dictionary implementation: indices start from 1 (because secondary terminals are in nodes 1 and 2 in this formulation)
        #highway_cost = road_cost_matrix[0][1 : num_terminals]
        #indices = list(range(1, num_terminals))
        #return dict(zip(indices, highway_cost))
