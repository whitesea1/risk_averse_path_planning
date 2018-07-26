#!/usr/bin/env python
import mdptoolbox, random
import numpy as np
from progressbar import ProgressBar
from copy import deepcopy
from math import sqrt
from time import time
import matplotlib.pyplot as plt
pbar = ProgressBar()
from scipy.spatial import Delaunay, distance


def _generate_spatial_graph(n = 100, blowout = 0.0, upper_bound = 100, max_var = 5.0, res = 1):
    # Use a Delaunay triangulation to make the graph, and then blow out random edges.
    # Use a percentage of edges to blow out. This is a percentage of redundant edges
    # in the graph, beyond the mimimun for connectivity
    np.random.seed(int(time()))
    size = float(res) * float(n)
    points = []
    nodes = dict()
    node_nums = range(n)
    for i in node_nums:
        nodes[i] = dict()
        points.append([size * np.random.rand(), size*np.random.rand()])
    points = np.array(points)
    graph = Delaunay(points)
    edges = []
    for simplex in graph.simplices:
        for i in range(len(simplex)):
            for j in range(i+1,len(simplex)):
                if (simplex[i],simplex[j]) not in edges and (simplex[j],simplex[i]) not in edges:
                    edges.append((simplex[i],simplex[j]))
                    nodes[simplex[i]][simplex[j]] = None
                    nodes[simplex[j]][simplex[i]] = None

    num_edges = len(edges)
    min_edges = n - 1
    blowout_length = int(blowout * (num_edges - min_edges))
    for i in range(blowout_length):
        removed = False
        while not removed:
            start = np.random.choice(node_nums)
            if not len(nodes[start]) > 1:
                # only one edge to that node left
                continue
            connections = [j for k,j in enumerate(nodes[start])]
            end = np.random.choice(connections)
            if not len(nodes[end]) > 1:
                # only one edge to the end node left
                continue
            else:
                nodes[start].pop(end,None)
                nodes[end].pop(start,None)
                removed = True
    for i in node_nums:
        for k,j in enumerate(nodes[i]):
            n_modes = np.random.choice(range(1,3))
            gaussians = []
            weight = 1.0
            start = points[i]
            end = points[j]
            low = distance.euclidean(start,end)
            for b in range(n_modes):
                if b < n_modes - 1:
                    w = np.random.random() * weight
                    weight = weight - w
                else:
                    w = weight
                mean = None
                var = None
                found = False
                while not found:
                    mean = (upper_bound - low)*np.random.random() + low
                    var = max_var*np.random.random()
                    if mean - (3*var) > low:
                        found = True
                gaussians.append((mean, var, w))
            nodes[i][j] = gaussians
    return(nodes,points)


# Code for the _bridge_util and bridge functions was found on https://www.geeksforgeeks.org/bridge-in-a-graph/
# and was modified for use in this structure


def _generate_graph(n = 100,k = None, low = 10, high = 100, max_var = 5.0):
    '''
        This function is going to generate a graph with n nodes and k edges.
        each edge is going to have anywhere between 1 and 3 components of a
        Gaussian Mixture model describing the transition cost.

        The return value of this function is a dictionary of dictionaries. The
        higher dictionary is keyed by the node, and the sub dictionaries are keyed
        by the node numbers for which there is an edge from the node. The values of
        the sub dictionaries are the gaussians that describe the transition cost.
    '''
    if k == None:
        k = np.random.choice(range(n - 1, n*2))
    if(k < n - 1) or k > n ** 2:
        print("k must be greater than or equal to n - 1 and less than n^2")
        return(1)
    nodes = dict()
    open_connections = []
    for i in range(n):
        nodes[i] = dict()
        for j in range(i):
            open_connections.append((i,j))
    for i in range(k):
        if i < n - 1:
            # first make sure connections are made to each node
            new = False
            while not new:
                j = np.random.choice(range(i,n))
                if open_connections.__contains__((j,i)):
                    new = True
            e = (j,i)
            open_connections.remove(e)
        else:
            random.shuffle(open_connections)
            e = open_connections.pop()
        n_modes = np.random.choice(range(1,3))
        gaussians = []
        weight = 1.0
        for j in range(n_modes):
            if j < n_modes - 1:
                w = np.random.random() * weight
                weight = weight - w
            else:
                w = weight
            gaussians.append(((high - low)*np.random.random() + low, max_var*np.random.random(), w))
        nodes[e[0]][e[1]] = gaussians
        nodes[e[1]][e[0]] = gaussians
    return(nodes)

def draw_graph(graph,points):
    plt.plot(points[:,0],points[:,1], 'o')
    drawn_lines = []
    for o,i in enumerate(graph):
        for k,j in enumerate(graph[i]):
            if not (i,j) in drawn_lines:
                drawn_lines.append((i,j))
                drawn_lines.append((j,i))
                plt.plot(points[[i,j],0],points[[i,j],1], 'r--', lw=2)
                plt.show(block = False)

def draw_policy(points, policy, goal, start):
    i = start
    while i != goal:
        plt.plot(points[[i,policy[i]],0],points[[i,policy[i]],1], 'b-', lw=4)
        i = policy[i]

def graph_to_mdp(graph,representation, goal):
    if representation == 'simple':
        '''
            Need to define the transitions and the rewards for each state. the state transition function
            will be of size S x S x A, where S is the number of states, and A is the number of actions (in this case)
            also of length, as moving to each node is an independent action, and not all actions can be taken at each state.
        '''
        n = len(graph)
        P = np.zeros((n,n,n))
        R = np.zeros((n,n))
        for j,i in enumerate(graph):
            edges = [k for l,k in enumerate(graph[i])]
            for q in range(n):
                if q in edges:
                    P[q,i,q] = 1.0
                    exp = sum([graph[i][q][e][0]*graph[i][q][e][2] for e in range(len(graph[i][q]))])
                    R[i,q] = -1.0*exp
                    # if q == goal:
                    #     R[i,q] = R[i,q]
                else:
                    P[q,i,i] = 1.0
                    R[i,q] = -1000
        R[goal,goal] = 0
        return P,R
    elif representation == 'multimodal':
        '''
            This is a much more complex representation. Here, each mode of a reward from the graph
            constitutes a new state. This means we can have up to 3x more states if the number of
            modes is limited to 3.
        '''
        keys = dict()
        n = len(graph)
        l = len(graph)
        # get the total number of states and intermediate states
        for i in range(len(graph)):
            for h,j in enumerate(graph[i]):
                for k in range(len(graph[i][j])):
                    keys[(i,j,k)] = l
                    l = l + 1
        P = np.zeros((n,l,l))
        R = np.zeros((n,l,l))
        for j,i in enumerate(graph):
            edges = [k for b,k in enumerate(graph[i])]
            for q in range(n):
                if q in edges:
                    # we have to define all the transitions
                    for e in range(len(graph[i][q])):
                        P[q,i,keys[(i,q,e)]] = graph[i][q][e][2]
                        R[q,i,keys[(i,q,e)]] = -1.0 * graph[i][q][e][0]
                        for w in range(n):
                            P[w,keys[(i,q,e)],q] = 1.0
                            R[w,keys[(i,q,e)],q] = 0.0
                else:
                    P[q,i,i] = 1.0
                    R[q,i,i] = -1000
        R[goal,goal,goal] = 0
        return P,R
    elif representation == 'simple_conservative':
        '''
            Need to define the transitions and the rewards for each state. the state transition function
            will be of size S x S x A, where S is the number of states, and A is the number of actions (in this case)
            also of length, as moving to each node is an independent action, and not all actions can be taken at each state.
            This mode also finds an assumed variance on the distribution, and uses the mean plus one standard deviation
            as a conservative estimate of cost.
        '''
        n = len(graph)
        P = np.zeros((n,n,n))
        R = np.zeros((n,n))
        for j,i in enumerate(graph):
            edges = [k for l,k in enumerate(graph[i])]
            for q in range(n):
                if q in edges:
                    P[q,i,q] = 1.0
                    samples = []
                    for t in range(1000):
                        for p in range(len(graph[i][q])):
                            samples.append(np.random.normal(graph[i][q][p][0],graph[i][q][p][1],1)[0])
                    exp = sum([graph[i][q][e][0]*graph[i][q][e][2] for e in range(len(graph[i][q]))]) + np.std(np.array(samples), axis=0)
                    R[i,q] = -1.0*exp
                    # if q == goal:
                    #     R[i,q] = R[i,q]
                else:
                    P[q,i,i] = 1.0
                    R[i,q] = -1000
        R[goal,goal] = 0
        return P,R
    elif representation == 'multimodal_conservative':
        '''
            This is a much more complex representation. Here, each mode of a reward from the graph
            constitutes a new state. This means we can have up to 3x more states if the number of
            modes is limited to 3. Here we are also assuming the cost of the edge is
            equal to the mean mean cost plus one standard deviation
        '''
        keys = dict()
        n = len(graph)
        l = len(graph)
        # get the total number of states and intermediate states
        for i in range(len(graph)):
            for h,j in enumerate(graph[i]):
                for k in range(len(graph[i][j])):
                    keys[(i,j,k)] = l
                    l = l + 1
        P = np.zeros((n,l,l))
        R = np.zeros((n,l,l))
        for j,i in enumerate(graph):
            edges = [k for b,k in enumerate(graph[i])]
            for q in range(n):
                if q in edges:
                    # we have to define all the transitions
                    for e in range(len(graph[i][q])):
                        P[q,i,keys[(i,q,e)]] = graph[i][q][e][2]
                        R[q,i,keys[(i,q,e)]] = -1.0 * (graph[i][q][e][0] + graph[i][q][e][1])
                        for w in range(n):
                            P[w,keys[(i,q,e)],q] = 1.0
                            R[w,keys[(i,q,e)],q] = 0.0
                else:
                    P[q,i,i] = 1.0
                    R[q,i,i] = -1000
        R[goal,goal,goal] = 0
        return P,R

def run_policy(graph,policy,initial):
    scores = []
    for i in range(1000):
        i = initial
        score = []
        while i != policy[i]:
            samples = []
            start = i
            end = policy[i]
            i = end
            for k in range(len(graph[start][end])):
                sampled = False
                n = None
                while not sampled:
                    n = np.random.normal(graph[start][end][k][0],graph[start][end][k][1],1)[0]
                    if n > 0:
                        sampled = True
                samples.append(n)

            score.append(np.random.choice(samples,p = [k[2] for k in graph[start][end]]))
        scores.append(sum(score))
    return scores

def get_path(policy, start, stop):
    path = [start]
    i = start
    while i != stop:
        i = policy[i]
        path.append(i)
    return path

def evaluate_method(graph, method, goal, start):
    P,R = graph_to_mdp(graph,method,goal)
    pi = mdptoolbox.mdp.ValueIteration(P,R,0.99, max_iter=1000)
    pi.run()
    scores = run_policy(graph,pi.policy, start)
    scores = np.array(scores)
    return np.mean(scores, axis=0), np.std(scores, axis = 0)

def compare_policies(graph,methods,goal):
    policies = []
    for method in methods:
        # print("testing method {}".format(method))
        P,R = graph_to_mdp(graph,method,goal)
        pi = mdptoolbox.mdp.ValueIteration(P,R,0.99, max_iter=1000)
        pi.run()
        policies.append(deepcopy(pi.policy))
    diffs = []
    for i in range(len(policies)):
        for j in range(i,len(policies)):
            for k in range(len(graph)):
                if policies[i][k] != policies[j][k]:
                    diffs.append(1)
                else:
                    diffs.append(0)
    return float(sum(diffs)) / float(len(diffs))


if __name__ == "__main__":
    # graph = _generate_graph(n = 100, k = 120)
    # P1,R1 = graph_to_mdp(graph,'simple',25)
    # pi1 = mdptoolbox.mdp.ValueIteration(P1,R1,0.99)
    # pi1.run()
    # scores1 = run_policy(graph,pi1.policy,1)
    # plt.subplot(2,2,1)
    # plt.ylabel('Simple')
    # n1,bins1,patches1 = plt.hist(scores1)
    #
    # P2,R2 = graph_to_mdp(graph,'multimodal',25)
    # pi2 = mdptoolbox.mdp.ValueIteration(P2,R2,0.99)
    # pi2.run()
    # scores2 = run_policy(graph,pi2.policy,1)
    # plt.subplot(2,2,2)
    # plt.ylabel('Multimodal')
    # n2,bins2,patches2 = plt.hist(scores2)
    #
    # P3,R3 = graph_to_mdp(graph,'simple_conservative',25)
    # pi3 = mdptoolbox.mdp.ValueIteration(P3,R3,0.99)
    # pi3.run()
    # scores3 = run_policy(graph,pi3.policy,1)
    # plt.subplot(2,2,3)
    # plt.ylabel('Simple_conservative')
    # n3,bins3,patches3 = plt.hist(scores3)
    #
    # P4,R4 = graph_to_mdp(graph,'multimodal_conservative',25)
    # pi4 = mdptoolbox.mdp.ValueIteration(P4,R4,0.99)
    # pi4.run()
    # scores4 = run_policy(graph,pi4.policy,1)
    # plt.subplot(2,2,4)
    # plt.ylabel('Multimodal_conservative')
    # n4,bins4,patches4 = plt.hist(scores4)
    #
    # print("State s_act m_act sc_act mc_act")
    # for i in range(len(graph)):
    #     print("{} {} {} {} {}".format(i,pi1.policy[i],pi2.policy[i],pi3.policy[i],pi4.policy[i]))
    #
    # print(get_path(pi1.policy,1,25))
    # print(get_path(pi2.policy,1,25))
    # print(get_path(pi3.policy,1,25))
    # print(get_path(pi4.policy,1,25))
    # print(get_path(pi1.policy,1,25) == get_path(pi2.policy,1,25))

    # plt.show()

    # Testing for sensitivity to parameters of the graph
    # variances = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    # size = [10, 15, 25, 50, 100, 150, 200, 300]
    # upper_bounds = [20,30,50,100,200]
    #
    # p1_var_means = []
    # p2_var_means = []
    # p3_var_means = []
    # p4_var_means = []
    #
    # p1_var_vars = []
    # p2_var_vars = []
    # p3_var_vars = []
    # p4_var_vars = []
    #
    # p1_size_means = []
    # p2_size_means = []
    # p3_size_means = []
    # p4_size_means = []
    #
    # p1_size_vars = []
    # p2_size_vars = []
    # p3_size_vars = []
    # p4_size_vars = []
    #
    # p1_bounds_means = []
    # p2_bounds_means = []
    # p3_bounds_means = []
    # p4_bounds_means = []
    #
    # p1_bounds_vars = []
    # p2_bounds_vars = []
    # p3_bounds_vars = []
    # p4_bounds_vars = []
    #
    # for i in variances:
    #     p1_means = []
    #     p1_vars = []
    #     p2_means = []
    #     p2_vars = []
    #     p3_means = []
    #     p3_vars = []
    #     p4_means = []
    #     p4_vars = []
    #     for j in range(10):
    #         graph = _generate_graph(n = 100, max_var = i)
    #         p1_mean, p1_var = evaluate_method(graph, 'simple', 1, 25)
    #         p2_mean, p2_var = evaluate_method(graph, 'simple_conservative', 1, 25)
    #         p3_mean, p3_var = evaluate_method(graph, 'multimodal', 1, 25)
    #         p4_mean, p4_var = evaluate_method(graph, 'multimodal_conservative', 1, 25)
    #         p1_means.append(p1_mean)
    #         p1_vars.append(p1_var)
    #         p2_means.append(p2_mean)
    #         p2_vars.append(p2_var)
    #         p3_means.append(p3_mean)
    #         p3_vars.append(p3_var)
    #         p4_means.append(p4_mean)
    #         p4_vars.append(p4_var)
    #     print("tested variance {}".format(i))
    #     p1_var_means.append(np.mean(np.array(p1_means)))
    #     p2_var_means.append(np.mean(np.array(p2_means)))
    #     p3_var_means.append(np.mean(np.array(p3_means)))
    #     p4_var_means.append(np.mean(np.array(p4_means)))
    #     p1_var_vars.append(np.mean(np.array(p1_vars)))
    #     p2_var_vars.append(np.mean(np.array(p2_vars)))
    #     p3_var_vars.append(np.mean(np.array(p3_vars)))
    #     p4_var_vars.append(np.mean(np.array(p4_vars)))
    #
    # for i in size:
    #     p1_means = []
    #     p1_vars = []
    #     p2_means = []
    #     p2_vars = []
    #     p3_means = []
    #     p3_vars = []
    #     p4_means = []
    #     p4_vars = []
    #     for j in range(10):
    #         graph = _generate_graph(n = i)
    #         p1_mean, p1_var = evaluate_method(graph, 'simple', 1, 9)
    #         p2_mean, p2_var = evaluate_method(graph, 'simple_conservative', 1, 9)
    #         p3_mean, p3_var = evaluate_method(graph, 'multimodal', 1, 9)
    #         p4_mean, p4_var = evaluate_method(graph, 'multimodal_conservative', 1, 9)
    #         p1_means.append(p1_mean)
    #         p1_vars.append(p1_var)
    #         p2_means.append(p2_mean)
    #         p2_vars.append(p2_var)
    #         p3_means.append(p3_mean)
    #         p3_vars.append(p3_var)
    #         p4_means.append(p4_mean)
    #         p4_vars.append(p4_var)
    #     print("tested size {}".format(i))
    #     p1_size_means.append(np.mean(np.array(p1_means)))
    #     p2_size_means.append(np.mean(np.array(p2_means)))
    #     p3_size_means.append(np.mean(np.array(p3_means)))
    #     p4_size_means.append(np.mean(np.array(p4_means)))
    #     p1_size_vars.append(np.mean(np.array(p1_vars)))
    #     p2_size_vars.append(np.mean(np.array(p2_vars)))
    #     p3_size_vars.append(np.mean(np.array(p3_vars)))
    #     p4_size_vars.append(np.mean(np.array(p4_vars)))
    #
    # for i in upper_bounds:
    #     p1_means = []
    #     p1_vars = []
    #     p2_means = []
    #     p2_vars = []
    #     p3_means = []
    #     p3_vars = []
    #     p4_means = []
    #     p4_vars = []
    #     for j in range(10):
    #         print("testing bound {}".format(i))
    #         graph = _generate_graph(high=i)
    #         p1_mean, p1_var = evaluate_method(graph, 'simple', 1, 9)
    #         p2_mean, p2_var = evaluate_method(graph, 'simple_conservative', 1, 9)
    #         p3_mean, p3_var = evaluate_method(graph, 'multimodal', 1, 9)
    #         p4_mean, p4_var = evaluate_method(graph, 'multimodal_conservative', 1, 9)
    #         p1_means.append(p1_mean)
    #         p1_vars.append(p1_var)
    #         p2_means.append(p2_mean)
    #         p2_vars.append(p2_var)
    #         p3_means.append(p3_mean)
    #         p3_vars.append(p3_var)
    #         p4_means.append(p4_mean)
    #         p4_vars.append(p4_var)
    #     p1_bounds_means.append(np.mean(np.array(p1_means)))
    #     p2_bounds_means.append(np.mean(np.array(p2_means)))
    #     p3_bounds_means.append(np.mean(np.array(p3_means)))
    #     p4_bounds_means.append(np.mean(np.array(p4_means)))
    #     p1_bounds_vars.append(np.mean(np.array(p1_vars)))
    #     p2_bounds_vars.append(np.mean(np.array(p2_vars)))
    #     p3_bounds_vars.append(np.mean(np.array(p3_vars)))
    #     p4_bounds_vars.append(np.mean(np.array(p4_vars)))
    #
    # plt.subplot(2,3,1)
    # plt.ylabel('Mean reward vs max_var')
    # plt.plot(variances,p1_var_means,variances,p2_var_means,variances,p3_var_means,variances,p4_var_means)
    # plt.legend(['s','s_c','m','m_c'])
    #
    # plt.subplot(2,3,2)
    # plt.ylabel('Variance vs max_var')
    # plt.plot(variances,p1_var_vars,variances,p2_var_vars,variances,p3_var_vars,variances,p4_var_vars)
    #
    # plt.subplot(2,3,3)
    # plt.ylabel('Mean reward vs size')
    # plt.plot(size, p1_size_means, size, p2_size_means, size, p3_size_means, size, p4_size_means)
    #
    # plt.subplot(2,3,4)
    # plt.ylabel('Variance vs size')
    # plt.plot(size, p1_size_vars, size, p2_size_vars, size, p3_size_vars, size, p4_size_vars)
    #
    # plt.subplot(2,3,5)
    # plt.ylabel('Mean reward vs upper_bound')
    # plt.plot(upper_bounds, p1_bounds_means, upper_bounds, p2_bounds_means, upper_bounds, p3_bounds_means, upper_bounds, p4_bounds_means)
    #
    # plt.subplot(2,3,6)
    # plt.ylabel('Variance vs upper_bound')
    # plt.plot(upper_bounds, p1_bounds_vars, upper_bounds, p2_bounds_vars, upper_bounds, p3_bounds_vars, upper_bounds, p4_bounds_vars)
    #
    # plt.show()

    # What conditions cause the different representations to produce different policies?
    #
    # variances = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    # size = [10, 15, 25, 50, 100, 150, 200, 300]
    # upper_bounds = [20,30,50,100,200, 300]
    #
    # policies = ['simple_conservative','multimodal_conservative']
    #
    # var_diff_rates = []
    #
    # size_diff_rates = []
    #
    # bound_diff_rates = []
    # n_graphs = 1
    # j = 0
    # for var in variances:
    #     rate = []
    #     for i in range(n_graphs):
    #         j = j+1
    #         print("{}%".format((float(j) * 100.)/float(n_graphs*len(variances) + n_graphs*len(size) + n_graphs*len(upper_bounds))))
    #         graph = _generate_graph(n = 100, max_var = var)
    #         rate.append(compare_policies(graph,policies,9))
    #     mean = sum(rate)
    #     mean = float(mean)/len(rate)
    #     var_diff_rates.append(mean)
    #
    # for s in size:
    #     rate = []
    #     for i in range(n_graphs):
    #         j = j+1
    #         print("{}%".format((float(j) * 100.)/float(n_graphs*len(variances) + n_graphs*len(size) + n_graphs*len(upper_bounds))))
    #         graph = _generate_graph(n = s)
    #         rate.append(compare_policies(graph,policies,9))
    #     mean = sum(rate)
    #     mean = float(mean)/len(rate)
    #     size_diff_rates.append(mean)
    #
    # for bound in upper_bounds:
    #     rate = []
    #     for i in range(n_graphs):
    #         j = j+1
    #         print("{}%".format((float(j) * 100.)/float(n_graphs*len(variances) + n_graphs*len(size) + n_graphs*len(upper_bounds))))
    #         graph = _generate_graph(high = bound)
    #         rate.append(compare_policies(graph,policies,9))
    #     mean = sum(rate)
    #     mean = float(mean)/len(rate)
    #     bound_diff_rates.append(mean)
    #
    # plt.subplot(3,1,1)
    # plt.ylabel('Changes from Variance diffs')
    # plt.plot(variances,var_diff_rates)
    #
    # plt.subplot(3,1,2)
    # plt.ylabel('Changes from size diffs')
    # plt.plot(size,size_diff_rates)
    #
    # plt.subplot(3,1,3)
    # plt.ylabel('Changes from Bounds diffs')
    # plt.plot(upper_bounds,bound_diff_rates)
    #
    # plt.show()

    # Let's try using graphs based on spatial coordinates
    # graph, points = _generate_spatial_graph(blowout = 0.95)
    # draw_graph(graph,points)
    # goal = 25
    # start = 0
    # P,R = graph_to_mdp(graph,'simple',goal)
    # pi = mdptoolbox.mdp.ValueIteration(P,R,0.99, max_iter=1000)
    # pi.run()
    # plt.pause(2)
    # draw_policy(points,pi.policy,goal,start)
    # plt.pause(20)
