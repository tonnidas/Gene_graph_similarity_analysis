# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import pickle
import os

from sknetwork.path.shortest_path import get_distances
from typing import Union, Optional
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from get_node_properties import *

# --------------------------------------------------------------------------------------------------------------
# Store graphs
# --------------------------------------------------------------------------------------------------------------

# Names of graph files (containing edges and their properties) inside "Input_Graphs" folder
graphs = ['timepoint0_baseline_graph', 'timepoint1_induction_graph', 'timepoint2_necropsy_graph']


# --------------------------------------------------------------------------------------------------------------
# Analysis between multiple graphs
# --------------------------------------------------------------------------------------------------------------

print("---------------------------------------------------------")
print('Graph edge property analysis')
print(edgePropertyAnalysis(graphs, True))

print("---------------------------------------------------------")
print('Graph node property analysis')
print(nodePropertyAnalysis(graphs, True))

# --------------------------------------------------------------------------------------------------------------
# Topological property level similarity and disimilarity
# --------------------------------------------------------------------------------------------------------------

# Store Chosen graphs
print("---------------------------------------------------------")
print('Store graph')
print(store_graph(graphs))

# Get basic properties of the graph
print("---------------------------------------------------------")
print('Graph properties: ')
print(getGraphProperties(graphs))

# Get basic properties of each node in a graph
graph_name = 'timepoint0_baseline_graph' 
print("---------------------------------------------------------")
print('Node properties: ')
print(getNodeProperties(graph_name))

# Get node property difference between two graphs
graph1, graph2 = 'timepoint0_baseline_graph', 'timepoint1_induction_graph'
print("---------------------------------------------------------")
print("Node property difference: ")
print(getDifference(graph1, graph2))

# Get value for each node property vector of a graph
graph = 'timepoint0_baseline_graph'
print("---------------------------------------------------------")
print("Value of the node property vector: ")
print(getStructureValue(graph))

# Get measurable node property-wise similar and disimilar nodes between two graphs
graph1, graph2 = 'timepoint0_baseline_graph', 'timepoint1_induction_graph'
print("---------------------------------------------------------")
print('Similar and disimilar Nodes: ')
similarAnddisimilarNodes(graph1, graph2, 0.1, 4)


# --------------------------------------------------------------------------------------------------------------
# Adjacency feature level similarity and disimilarity
# --------------------------------------------------------------------------------------------------------------

# Get edge property vector of a graph
graph = 'timepoint0_baseline_graph'
print("---------------------------------------------------------")
print("Edge property vector: ")
print(getEdgeProperties(graph))

# Get difference in values of edge property vector of two graphs
graph1, graph2 = 'timepoint0_baseline_graph', 'timepoint1_induction_graph'
print("---------------------------------------------------------")
print('Edge property difference vector: ')
print(getEdgePropertyDifferenceDf(graph1, graph2))