#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, distance
from time import time
import mdptoolbox

# '''
#     This class has functions to automatically generate random graphs
#     embedded in a n-dimensional space. The costs of edges
#     in the graph will be represented by a set of weighted means and
#     standard deviations. This is meant to represent distinct types of
#     errors that cannot necesarrily be predicted before traversal of an edge
#     but can be observed after many traversals of that edge.
#
#     Class variables:
#         time: variable used for the method bridge
#         bridges: list of bridges in the graph after the call of bridges
#         V: the number of vertices in the graph
#         graph: the dictionary representing the connections to each node
#             The graph has the following structure:
#             {v: {e: [(mean,std_dev,weight)], e: []}, v: {e: []}}
#             where v is the vertice and e is an edge from that vertice
#             and mean is the mean of one gaussian, std_dev is the standard
#             deviation of that gaussian, and weight is the weight of that
#             gaussian. The sum of weights for each gaussian in an edge must be
#             1
#         points: the coordinates of each point in the graph
#         upper_bound: the maximum mean cost that will placed on an edge
#         max_var: The maximum std_dev that a guassian describing an edge cost can have
#         num_modes: the maximum number of gaussians describing an edge cost
# '''
class multimodal_graph:
    def __init__(self):
        self.time = 0
        self.bridges = None
        self.V = None
        self.graph = dict()
        self.points = []
        self.upper_bound = 100
        self.max_var = 0.0
        self.num_modes = 3
        self.dims = None

    '''
        The generate function will generate the graph based on the parameters passed
        to the function. This function will call the class functions add_edge,
        remove_edge, and bridge

        Parameters:
        vertices: the number of vertices to generate
        dims: the upper limits of the dimensions in which the points will be sampled
        blowout: the ratio of edges that are redundant that will be blown out
        upper_bound: the maximum mean cost that will placed on an edge
        max_var: The maximum std_dev that a guassian describing an edge cost can have
        num_modes: the maximum number of gaussians describing an edge cost
    '''
    def generate(self,vertices = 100, dims = [100,100], blowout = 0.5, upper_bound = 100, max_var = 5.0, num_modes = 3):
        self.graph = dict()
        np.random.seed(int(time()))
        self.V = vertices
        self.upper_bound = upper_bound
        self.max_var = max_var
        self.num_modes = num_modes
        self.dims = len(dims)

        for i in range(self.V):
            self.graph[i] = dict()
            self.points.append([float(j) * np.random.rand() for j in dims])

        self.points = np.array(self.points)

        # Generate a triangulated graph of the points generated, and randomly
        # blowout edges while maintaining the connectivity of the graph
        tri = Delaunay(self.points)
        edges = []
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    if ((simplex[i],simplex[j]) not in edges and (simplex[j],simplex[i]) not in edges):
                        edges.append((simplex[i],simplex[j]))
                        self.graph[simplex[i]][simplex[j]] = None
                        self.graph[simplex[j]][simplex[i]] = None
        num_edges = len(edges)
        min_edges = self.V - 1
        blowout_length = int(blowout * (num_edges - min_edges))
        for i in range(blowout_length):
            removed = False
            while not removed:
                self._bridge()
                start = np.random.choice(range(self.V))
                end = np.random.choice([j for k,j in enumerate(self.graph[start])])
                if (start,end) in self.bridges or (end,start) in self.bridges:
                    continue
                else:
                    self.graph[start].pop(end,None)
                    self.graph[end].pop(start,None)
                    removed = True


        # Generate costs for each of the edges in the graph
        for k,i in enumerate(self.graph):
            for l,j in enumerate(self.graph[i]):
                self.add_edge(i,j)

    '''
        _bridge stores the list of edges that are bridges, so that they do not get
        broken in the blowout section of the generate function. This code was found
        on https://www.geeksforgeeks.org/bridge-in-a-graph/
    '''
    def _bridge(self):

        # Mark all the vertices as not visited and Initialize parent and visited,
        # and ap(articulation point) arrays
        self.time = 0
        self.bridges = []
        visited = [False] * (self.V)
        disc = [float("Inf")] * (self.V)
        low = [float("Inf")] * (self.V)
        parent = [-1] * (self.V)

        # Call the recursive helper function to find bridges
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if visited[i] == False:
                self._bridgeUtil(i, visited, parent, low, disc)

    '''
        _bridge_util is a recursive function used by _bridge. It was also found
        on https://www.geeksforgeeks.org/bridge-in-a-graph/
    '''
    def _bridgeUtil(self, u, visited, parent, low, disc):

        #Count of children in current node
        children =0

        # Mark the current node as visited
        visited[u]= True

        # Initialize discovery time and low value
        disc[u] = self.time
        low[u] = self.time
        self.time += 1

        #Recur for all the vertices adjacent to this vertex
        for i,v in enumerate(self.graph[u]):
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[v] == False :
                parent[v] = u
                children += 1
                self._bridgeUtil(v, visited, parent, low, disc)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[u] = min(low[u], low[v])


                ''' If the lowest vertex reachable from subtree
                under v is below u in DFS tree, then u-v is
                a bridge'''
                if low[v] > disc[u]:
                    self.bridges.append((u,v))


            elif v != parent[u]: # Update low value of u for parent function calls.
                low[u] = min(low[u], disc[v])

    '''
        add_edge will add a directed edge from i to j as long as i and j are valid
        vertices in the graph
    '''
    def add_edge(self,i,j):
        if i > self.V or j > self.V:
            raise ValueError('vertices that do not exist cannot be connected', i, j)
        else:
            self.graph[i][j] = self._gen_cost(i,j)

    '''
        _gen_cost will return a list of means and std_dev describing the
        cost between two nodes. Costs are defined such that no samples within
        3 standard deviations below the mean will be lower than euclidean distance
        between the points
    '''
    def _gen_cost(self,i,j):
        gaussians = []
        weight = 1.0
        start = self.points[i]
        end = self.points[j]
        low = distance.euclidean(start,end)
        for i in range(self.num_modes):
            if j < self.num_modes - 1:
                w = np.random.random() * weight
                weight = weight - w
            else:
                w = weight
            mean = None
            var = None
            found = False
            while not found:
                mean = (self.upper_bound - low)*np.random.random() + low
                var = self.max_var*np.random.random()
                if (mean - 3*var) > low:
                    found = True
            gaussians.append((mean,var,w))
        return gaussians

    '''
        get_graph returns the graph
    '''
    def get_graph(self):
        return self.graph

    '''
        get_points returns the coordinates of the vertices in the graph
    '''
    def get_points(self):
        return self.points

    '''
        draw_graph uses pyplot to draw the graph with the current edges, though it
        does not automatically call the plt.show() function. This function only
        works for 2d graphs at the moment
    '''
    def draw_graph(self):
        if self.dims > 2 or self.dims < 2:
            raise ValueError('draw_graph only works with 2d graphs')

        plt.plot(self.points[:,0],self.points[:,1], 'o')
        drawn_lines = []
        for o,i in enumerate(self.graph):
            for k,j in enumerate(self.graph[i]):
                if not (i,j) in drawn_lines:
                    drawn_lines.append((i,j))
                    drawn_lines.append((j,i))
                    plt.plot(self.points[[i,j],0],self.points[[i,j],1], 'r--', lw=.5)

    '''
        generate_mdp returns a transition probability matrix and a reward
        function for the graph, based on the representation provided.
        The options for representation are:

        simple: Edge costs are modeled as a mean of the gaussians in the edge
        simple_conservative: Edge costs are modeled as a mean of the gaussians,
            plus one sample standard deviation derived from sampling from
            the underlying gaussians 1000 times
        multimodal: Additional vertices are modeled as intermediate steps
            along an edge, and are treated as a stochastic outcome
            of the action of traversing the edge. These nodes form directed
            along the edge they were derived from. In this case the cost of
            traversing the intermediate edge to one of these hidden vertices is
            the mean of the gaussian that describes that component
        multimodal_conservative: This is just like multimodal, with the only
            difference being that the cost of going to an intermediate vertice
            is assumed to be the mean of the gaussian desribing the cost plus
            one standard deviation

        The function also expects a goal state to move to, which must be a
        vertice on the graph
    '''
    def generate_mdp(self, representation, goal):
        graph = self.graph
        '''
        Need to define the transitions and the rewards for each state. the state transition function
        will be of size S x S x A, where S is the number of states, and A is the number of actions (in this case)
        also of length, as moving to each node is an independent action, and not all actions can be taken at each state.
        '''
        if representation == 'simple':
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

    '''
        draw_policy will take a list representing actions to take
        from each action, a starting vertice, and an ending vertice
        and it will draw the path taken following the policy
    '''
    def draw_policy(self,policy, start, goal, lnspec = '-b'):
        i = start
        while i != goal:
            plt.plot(self.points[[i,policy[i]],0],self.points[[i,policy[i]],1], lnspec, lw=4)
            i = policy[i]

    '''
        simulate_policy takes a policy, a start point, and a number of
        simulations to run, and returns a list of total rewards received
        executing the policy on the actual graph
    '''
    def simulate_policy(self,policy,start,runs = 1000):
        scores = []
        for i in range(runs):
            i = initial
            score = []
            while i != policy[i]:
                samples = []
                start = i
                end = policy[i]
                i = end
                for k in range(len(self.graph[start][end])):
                    sampled = False
                    n = None
                    while not sampled:
                        n = np.random.normal(self.graph[start][end][k][0],self.graph[start][end][k][1],1)[0]
                        if n > 0:
                            sampled = True
                    samples.append(n)

                score.append(np.random.choice(samples,p = [k[2] for k in self.graph[start][end]]))
            scores.append(sum(score))
        return scores

if __name__ == "__main__":
    graph = multimodal_graph()
    graph.generate(vertices = 5, blowout = 0.0)
    graph.draw_graph()
    plt.show(block = False)
    plt.pause(2)

    goal = 3
    start = 0


    P1,R1 = graph.generate_mdp('simple',goal)
    pi1 = mdptoolbox.mdp.ValueIteration(P1,R1,0.99, max_iter=1000)
    pi1.run()
    graph.draw_policy(pi1.policy,start,goal, 'b-')
    # plt.pause(2)

    P2,R2 = graph.generate_mdp('simple_conservative',goal)
    pi2 = mdptoolbox.mdp.ValueIteration(P2,R2,0.99, max_iter=1000)
    pi2.run()
    graph.draw_policy(pi2.policy,start,goal, 'g-')
    # plt.pause(2)

    P3,R3 = graph.generate_mdp('multimodal',goal)
    for i in range(len(P3)):
        for j in range(len(P3)):
            if sum(P3[i][j]) < 1.0:
                print sum(P3[i][j])
                print i,j
                print P3[i][j]
    print graph.get_graph()
    pi3 = mdptoolbox.mdp.ValueIteration(P3,R3,0.99, max_iter=1000)
    pi3.run()
    graph.draw_policy(pi3.policy,start,goal, 'c-')
    plt.pause(2)

    P4,R4 = graph.generate_mdp('multimodal_conservative',goal)
    pi4 = mdptoolbox.mdp.ValueIteration(P4,R4,0.99, max_iter=1000)
    pi4.run()
    graph.draw_policy(pi4.policy,start,goal, 'k-')
    plt.pause(2)

    plt.pause(20)
