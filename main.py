import networkx as nx
import sys

#create a graph class and store the graph in it
#weights should be stored with the nodes in the adjacency list, stored as [node edge goes to, weight].
class graph(object):

  def __init__(self, graphDict = None, calculate = False, MST = None, O = None, M = None, EM = None, ET = None, HP = None):
    if graphDict == None:
      self.graphDict = {}
    else:
      # {'A': [['B', 3], ['C',2]..] 'B': [['A', 3], ['C', 4]...] }
      self.graphDict = graphDict
    if calculate:
      # We iteratively calculate the important sub-graphs of G
      # Chaining them off previous ones and storing as
      # attrs, all graphs are instances of the graph object
      self.MST = self.MST()
      self.O = self.oddVertices()
      self.M = self.minWeightPerfectMatching()
      self.EM = self.eulerianMultigraph()
      self.ET = self.eulerianTour()
      self.HP = self.eulerianToHamiltonian()

  '''
  Method
  Input: self
  Output: the list of vertices in the graph in the form [vertex,vertex,...]
  Approach: we simply use the built in function .keys() in 
  order to access the keys in the graph, which are the vertices
  '''
  def getVertices(self):
    return list(self.graphDict.keys())
  
  '''
  Method
  Input: self
  Output: the list of edges in the graph in the form [(vertex,vertex,weight),...]
  Approach: We iterate through each vertex and then through each edge connected to that vertex and append that edge to the list edges.
  Runtime: O(|E|)
  '''
  def getEdges(self):
    edges = []

    # We have to unpack the adjacency list into
    # more usable form with whole edges
    for key in self.graphDict:
      for e in self.graphDict[key]:
        edges.append((key, e[0], e[1]))

    # Sort by edge weight stored in index 2
    edges.sort(key = lambda x: x[2])

    # Returns in form [('A', 'B', 3), ('B', 'A', '3')...]
    return edges

  '''
  Method
  Input: self, vertices u and v
  Output: boolean, are u and v in a connected component]
  Approach: Here, we use Breadth First search starting at u. If v is visited by the time we finish traversing the graph, then the u and v are in the same connected component. 
  Runtime: O(|E|+|V|)
  '''
  def connected(self, u, v):
    # Implementation of BFS to determine whether
    # u and v are connected in this graph
    visited = [u]
    queue = [u]

    while queue:
      s = queue.pop(0)
      for neighbor in self.graphDict[s]:
        if neighbor[0] not in visited:
          visited.append(neighbor[0])
          queue.append(neighbor[0])
    
    if v in visited:
      return True
    return False

  '''
  Method
  Input: self
  Output: a graph object containing the minimum spanning tree of the graph.
  Approach: We utilize Kruskall's Algorithm where we add the next smallest edge that connects two previously unconnected components
  Runtime: O(|E|(|E|+|V|)) = O(|E|^2)
  '''
  def MST(self):
    # Implementing Kruskall's algorithm
    T = graph()
    for v in self.getVertices():
      T.graphDict[v] = []
    
    # Edges are already sorted!
    edges = self.getEdges()
    while edges:
      e = edges.pop(0)
      if not T.connected(e[0], e[1]):
        # To make T a graph object, we need to 
        # reorganize edges into adjacency list
        T.graphDict[e[0]]  = T.graphDict[e[0]] + [[e[1], e[2]]]
        T.graphDict[e[1]]  = T.graphDict[e[1]] + [[e[0], e[2]]]

    return T

  '''
  Method
  Input: self
  Output: a graph object containing the vertices with odd edge degrees in the MST
  Approach: We iterate through each vertex in the MST and count the number of edges in the MST that connect to that vertex. We then construct a graph that connects all of these add vertices
  Runtime: O(|E|)
  '''
  def oddVertices(self):
    O_nodes = []
    for node in self.MST.getVertices():
      # Adjacency list allows fast count of edge degree
      if len(self.MST.graphDict[node]) % 2 != 0:
        O_nodes.append(node)
    
    # Find the subset of edges where both the
    # head and tail have odd degree
    # Adjacency list format
    O_adjlist = {}
    for v in O_nodes:
      for e in self.graphDict[v]:
        if e[0] in O_nodes and e not in self.MST.graphDict[v]:
          O_adjlist[v] = O_adjlist.get(v, []) + [[e[0], e[1]]]

    O = graph(O_adjlist)
    return O

  '''
  Input: self
  Output: returns a minimum weight perfect matching in the graph returned by the previous method.
  Approach: we use a built in maximum weight matching algorithm and simply invert the edge weight before we feed it into this algorithm.
  Runtime: We assume that the ".max_weight_matching(G, maxcardinality = False, weight = "weight")" method requires O(|V|^3) time since the runtime is not explicitly stated in the documentation. Thus, the total runtime of this method will be O(|V|^3)
  '''
  def minWeightPerfectMatching(self):
    # Using built-in packages requires us to shift
    # graph into networkx compatible structure 
    # We take the reciporical of the edge weight
    # to turn the max_weight_matching into
    # min weight matching
    G = nx.Graph()
    for e in self.O.getEdges():
      G.add_edge(e[0], e[1], weight = 1 / e[2])
    
    #Built-in networkx function
    # Returns in form [('A', 'B'), ('C', 'D')]
    M_set = nx.algorithms.matching.max_weight_matching(G, maxcardinality = False, weight = "weight")
    
    # Rebuild as adajacency list graph
    M = {}
    for pair in M_set:
      M[pair[0]] = M.get(pair[0], []) + [[pair[1], 1 / G[pair[0]][pair[1]]['weight']]]
      M[pair[1]] = M.get(pair[1], []) + [[pair[0], 1 / G[pair[0]][pair[1]]['weight']]]
    return graph(M)

  '''
  Input: self
  Output: a graph object containing the combination of the perfect matching and the MST
  Approach: Here, we simply append the edges to the adjacency list and check to make sure that we never append an edge twice.
  Runtime:O(|E|)
  '''
  def eulerianMultigraph(self):
    # Merge the adjacency lists of the MST
    # and M to get the multigraph
    EM_adjlist = self.MST.graphDict
    for v in self.M.getVertices():
      for e in self.M.graphDict[v]:
        if e not in EM_adjlist[v]:
          EM_adjlist[v] = EM_adjlist[v] + [e]

    return graph(EM_adjlist)
    
  '''
  Input: self
  Output: A list of vertices that visits every edge exactly once
  Approach: We simply implement the built in ".eulerian_circuit(G)" method that implements Hierholzer’s algorithm.
  Runtime:O(|E|) which is the runtime of Hierholzer’s algorithm
  '''
  def eulerianTour(self):
    # Hierholzer’s algorithm
    # Again we use built-in networkx functions
    G = nx.Graph()
    for e in self.EM.getEdges():
      G.add_edge(e[0], e[1], weight = e[2])
      G.add_edge(e[1], e[0], weight = e[2])
    Tour = [u for u, v in nx.eulerian_circuit(G)]
    Tour.append(Tour[0])

    # The tour returned is a list of vertices, not a graph object
    return Tour
  
  '''
  Input: self
  Output: A list of vertices that comprise a Hamiltonian tour
  Approach: Here, we simply create a dictionary indexed by vertices called visited where we store whether or not we have already visited a node in our eurlerian circuit. If we have, then we simply skip that node until we arrive at the starting node or we encounter a node we have not yet visited.
  Runtime: O(|E|) since we need to run through every node in the Eulerian Circuit, which visits every edge in the multigraph
  '''
  def eulerianToHamiltonian(self):
    #Traverse the Eulerian circuit in order to find the nodes
    
    visited = {}
    for v in self.getVertices():
      visited[v] = False

    maxsize = len(self.ET)
    Hamiltonian = [chr]*maxsize

    # We iterate through the vertex list
    # if visited, we merely skip the vertex
    # and don't append it to the path
    j = 0
    for i in range(maxsize):
      if (visited[self.ET[i]] == False):
        Hamiltonian[j] = self.ET[i]
        visited[self.ET[i]] = True
        j+=1

    Hamiltonian[j] = Hamiltonian[0]
    j+=1

    #We only return the list of nodes in the Hamiltonian Circuit
    if j < maxsize:
      Hamiltonian = Hamiltonian[:j]

    return Hamiltonian

'''
Input: a text file
Output: An adjacency list in the form {vertex: [other node, weight]}
Approach: split each line of the file into an edge and append this 
edge to the appropriate adjacency list.
Runtime: O(|E|) since we encounter each edge once and we assume the graph is fully connected.
'''
def fileToAdjList(file_name):
  with open(file_name, "r") as f:
    lines = f.readlines()
    
    # Creates adjacency list out of edges 
    adjlist = {}
    for ln in lines:
      edge = ln.strip().split(",")
      adjlist[edge[0]] = adjlist.get(edge[0], []) + [[edge[1], float(edge[2])]]
      adjlist[edge[1]] = adjlist.get(edge[1], []) + [[edge[0], float(edge[2])]]
  
  # {'A': [['B', 3], ['C',2]..] 'B': [['A', 3], ['C', 4]...] }
  return adjlist

if __name__ == "__main__":
  try:
    Gadjlist = fileToAdjList(sys.argv[1])
  except:
    print("You either didn't type in an input file or your input file could not be converted to a graph. Please try again with a valid text file.")
    sys.exit()
  
  try:
    G = graph(Gadjlist, calculate = True)
  except:
    print("Please try again with a text file in the correct format.")
    sys.exit()
  
  print("Graph:", G.graphDict)

  #1: Find the Minimum Spanning Tree through Kruskall's Algorithm
  #print("MST:", G.MST.graphDict)

  #2: Find the Vertices with odd degree in the Minimum Spanning Tree
  print("Odd Verticies:", G.O.graphDict)

  #2: Find the perfect matching between vertices with odd degree using the Blossum algorithm 
  print("Perfect Matching of G:", G.M.graphDict)

  #3: Combine the edges in the MST and the perfect matching between vertices with odd degree
  print("Multigraph:", G.EM.graphDict)

  #4: Find the Eulerian Circuit of the Minimum spanning tree combined with the perfect matching between vertices with odd degree through the Hierholzer’s algorithm 
  print("Eulerian Tour:", G.ET)
  
  #4 Convert the Eulerian Circuit to a Hamiltonian Circuit
  print("Hamiltonian Circuit:", G.HP)



  
  
  