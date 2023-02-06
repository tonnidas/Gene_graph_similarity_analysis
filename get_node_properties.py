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

# ------------------------------------------------------------------------------------------------------        
# Helping method
# ------------------------------------------------------------------------------------------------------

def get_graph(G_name):
    file_pickle = 'Graph_Pickles/{}.pickle'.format(G_name)
    with open(file_pickle, 'rb') as handle: G = pickle.load(handle) 
    return G


# ------------------------------------------------------------------------------------------------------        
# Topological Property
# ------------------------------------------------------------------------------------------------------
def store_graph(graphs):

    for each_graph in graphs:
        file = 'Input_Graphs/{}.txt'.format(each_graph)  
        with open(file) as f: lines = f.readlines()

        edgeList = list()
        for line in lines:
            s = line.split()
            edgeList.append((s[0], s[1]))

        graph = nx.from_edgelist(edgeList)
        file_pickle = 'Graph_Pickles/{}.pickle'.format(each_graph)
        with open(file_pickle, 'wb') as handle: pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return "Graphs are stored inside 'Graph_Pickle' folder "
# ------------------------------------------------------------------------------------------------------


# ----------------------------------------
def getGraphProperties(G_list):

    graphs_list = []
    for file_name in G_list:
        file_pickle = 'Graph_Pickles/{}.pickle'.format(file_name)
        with open(file_pickle, 'rb') as handle: graph = pickle.load(handle) 
        graphs_list.append(graph)

    graphProperties = pd.DataFrame(columns=['graph_name', 'is_connected?', 'components_number', 'total_nodes', 'total_edges'])

    i = 0
    for each_graph in graphs_list:
        graphProperties.loc[len(graphProperties.index)] = [G_list[i], nx.is_connected(each_graph), nx.number_connected_components(each_graph), each_graph.number_of_nodes(), each_graph.number_of_edges()]
        i += 1

    graphProperties = graphProperties.set_index('graph_name')
    return graphProperties
# ----------------------------------------


# ----------------------------------------
def getNodeProperties(G_name):
    
    file_pickle = 'Graph_Pickles/{}.pickle'.format(G_name)
    with open(file_pickle, 'rb') as handle: G = pickle.load(handle) 

    # print('calculating degree')
    temp_dict = dict(G.degree)
    temp_max = max(temp_dict.values())
    degree_regularized = {key: value / temp_max for key, value in temp_dict.items()}
    df1 = pd.DataFrame.from_dict(degree_regularized, orient='index', columns=['degree'])

    # print('calculating degree_centrality')
    df2 = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['degree_centrality'])

    # print('calculating clustering_coefficient')
    df3 = pd.DataFrame.from_dict(nx.clustering(G), orient='index', columns=['clustering_coefficient'])

    # print('calculating eccentricity')
    graphs = list(G.subgraph(c) for c in nx.connected_components(G)) # returns a list of disconnected graphs as subgraphs
    dict_1 = {}
    for subgraph in graphs:
        dict_2 = nx.eccentricity(subgraph)
        dict_1 = {**dict_1,**dict_2}
    max_ecc = max(dict_1.values())
    ecc_regularized = {key: value / max_ecc for key, value in dict_1.items()}
    df4 = pd.DataFrame.from_dict(ecc_regularized, orient='index', columns=['eccentricity'])

    # print('calculating closeness_centrality')
    df5 = pd.DataFrame.from_dict(nx.closeness_centrality(G), orient='index', columns=['closeness_centrality'])

    # print('calculating betweenness_centrality')
    df6 = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index', columns=['betweenness_centrality'])

    # print('done calculating node properties')
    features = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
    print(">>>>>>>>>>>")
    print(features)
    print("======")
    features.index.names = ['node_name']
    print(features)
    return features
# ----------------------------------------

# ----------------------------------------
def getDifference(graph1, graph2):

    if get_graph(graph1).number_of_nodes() > get_graph(graph2).number_of_nodes(): nodes = list(get_graph(graph2).nodes)  
    elif get_graph(graph1).number_of_nodes() < get_graph(graph2).number_of_nodes(): nodes = list(get_graph(graph1).nodes) 
    else: nodes = nodes = list(get_graph(graph1).nodes) 

    node_names = nodes

    df1 = getNodeProperties(graph1)
    df2 = getNodeProperties(graph2)

    differenceDf = pd.DataFrame(columns=['node_name', 'degree', 'clustering_coefficient', 'eccentricity', 'closeness_centrality', 'betweenness_centrality'])
    
    for key in node_names:
        
        # differenceDf.loc[]
        # differenceDf.loc[key]['degree'] = (abs(df1.loc[key]['degree'] - df2.loc[key]['degree']))
        degree = abs(df1.loc[key]['degree'] - df2.loc[key]['degree'])
        clus = abs(df1.loc[key]['clustering_coefficient'] - df2.loc[key]['clustering_coefficient'])
        ec = abs(df1.loc[key]['eccentricity'] - df2.loc[key]['eccentricity'])
        close = abs(df1.loc[key]['closeness_centrality'] - df2.loc[key]['closeness_centrality'])
        be = abs(df1.loc[key]['betweenness_centrality'] - df2.loc[key]['betweenness_centrality'])
        differenceDf.loc[len(differenceDf.index)] = [key, degree, clus, ec, close, be]

    differenceDf = differenceDf.set_index('node_name')
    return differenceDf
# ----------------------------------------

# ----------------------------------------
def getStructureValue(graph):
    df1 = getNodeProperties(graph)
    differenceDf = pd.DataFrame(columns=['node_name', 'change'])

    for key, row in df1.iterrows():
        # print(key, row['degree'] , row['clustering_coefficient'] , row['eccentricity'] , row['closeness_centrality'] , row['betweenness_centrality'])
        differenceDf.loc[len(differenceDf.index)] = [key, (row['degree'] + row['clustering_coefficient'] + row['eccentricity'] + row['closeness_centrality'] + row['betweenness_centrality'])]

    differenceDf = differenceDf.set_index('node_name')
    return differenceDf
# ----------------------------------------

# ----------------------------------------
# Find out which nodes have below 0.5 change (in scale 0 to 5) from sdf
def similarAnddisimilarNodes(g1, g2, similarityThreshold, disimilarityThreshold):
    with open('Graph_Pickles/{}.pickle'.format(g1), 'rb') as handle: G1 = pickle.load(handle) 
    with open('Graph_Pickles/{}.pickle'.format(g2), 'rb') as handle: G2 = pickle.load(handle) 

    if len(set(list(G1.nodes)) - set(list(G2.nodes))) > 0:
        print("Node", set(list(G1.nodes)) - set(list(G2.nodes)), "of", g1, "are not present in", g2)  # Like {'Deoxycorticosterone', 'Aldosterone'}
    elif len(set(list(G2.nodes)) - set(list(G1.nodes))) > 0:
        print("Node", set(list(G2.nodes)) - set(list(G1.nodes)), "of", g2, "are not present in", g1)  # Like {'Deoxycorticosterone', 'Aldosterone'}
    else: print(g1, g2, "have the same nodes")

    if G1.number_of_nodes() > G2.number_of_nodes(): nodes = list(G2.nodes)  
    elif G1.number_of_nodes() < G2.number_of_nodes(): nodes = list(G1.nodes) 
    else: nodes = nodes = list(G1.nodes) 
    gdf = getDifference(g1, g2)

    tempDf = pd.DataFrame(columns=['node_name', 'change'])
    for key, row in gdf.iterrows():
        tempDf.loc[len(tempDf.index)] = [key, (row['degree'] + row['clustering_coefficient'] + row['eccentricity'] + row['closeness_centrality'] + row['betweenness_centrality'])]

    tempDf = tempDf.set_index('node_name')
    sdf = tempDf
    
    similar = []
    disimilar = []
    for key in nodes:
        if sdf.loc[key]['change'] == similarityThreshold: similar.append(key)
        if sdf.loc[key]['change'] > disimilarityThreshold: disimilar.append(key)
    print("Similar nodes between these graphs =", similar)
    print("disimilar nodes between these graphs =", disimilar)
# ----------------------------------------

# ------------------------------------------------------------------------------------------------------        
# Adjacency Property
# ------------------------------------------------------------------------------------------------------

# ----------------------------------------
def getEdgeProperties(file_name):
    with open('Input_Graphs/{}.txt'.format(file_name)) as f: lines = f.readlines()
    edgePropertiesdf = pd.DataFrame(columns=['edge_name', 'unsigned_weighted_rho', 'weighted_rho', 'rho'])
    
    for line in lines:
        s = line.split()

        edge_name = (s[0], s[1])
        unsigned_weighted_rho = float(s[3][:-1])
        weighted_rho = float(s[5][:-1])
        rho = float(s[7][:-1])

        edgePropertiesdf.loc[len(edgePropertiesdf.index)] = [edge_name, unsigned_weighted_rho, weighted_rho, rho]

    edgePropertiesdf = edgePropertiesdf.set_index('edge_name')
    return edgePropertiesdf
# ----------------------------------------

# ----------------------------------------
def getEdgePropertyDifferenceDf(G1_name, G2_name):
    G1_edgePropertiesDf = getEdgeProperties(G1_name)
    G2_edgePropertiesDf = getEdgeProperties(G2_name)
    with open('Graph_Pickles/{}.pickle'.format(G1_name), 'rb') as handle: G1 = pickle.load(handle) 
    with open('Graph_Pickles/{}.pickle'.format(G2_name), 'rb') as handle: G2 = pickle.load(handle) 

    indexsG1 = G1_edgePropertiesDf.index.values.tolist()     # (pd.Index(G1_edgePropertiesDf)).to_list()
    indexsG2 = G2_edgePropertiesDf.index.values.tolist()     # (pd.Index(G2_edgePropertiesDf)).to_list()

    if len(set(indexsG1) - set(indexsG2)) > 0:
        print("Total", len(set(indexsG1) - set(indexsG2)), "edges of", G1_name, "are not present in", G2_name)  
    if len(set(indexsG2) - set(indexsG1)) > 0:
        print("Total", len(set(indexsG2) - set(indexsG1)), "edges of", G2_name, "are not present in", G1_name)  

    edgePropertiesDifferencedf = abs(G1_edgePropertiesDf.subtract(G2_edgePropertiesDf))
    print(">>>>>>>>>>>")
    print(edgePropertiesDifferencedf.dropna())
    print("======")

    return edgePropertiesDifferencedf
# ----------------------------------------

# ----------------------------------------
def edgePropertyAnalysis(G_list, excel_store):

    df_final = pd.DataFrame({'edge_name': pd.Series(dtype='str')})

    for each_graph in G_list:
        suffix = '_' + each_graph.split('_')[0]
        df = getEdgeProperties(each_graph)
        df = df.add_suffix(suffix).reset_index()
        df_final = df_final.merge(df, on = 'edge_name', suffixes = [None, suffix], how='outer')

    if excel_store == True:
        df_final.to_excel('edgePropertyAnalysis.xlsx') 
    
    return df_final
# ----------------------------------------

# ----------------------------------------
def nodePropertyAnalysis(G_list, excel_store):

    df_final = pd.DataFrame({'node_name': pd.Series(dtype='str')})

    for each_graph in G_list:
        suffix = '_' + each_graph.split('_')[0]
        df = getNodeProperties(each_graph)
        print(df)
        df = df.add_suffix(suffix).reset_index()
        df_final = df_final.merge(df, on = 'node_name', suffixes = [None, suffix], how='outer')

    if excel_store == True:
        df_final.to_excel('nodePropertyAnalysis.xlsx') 

    return df_final
# ----------------------------------------