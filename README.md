In this repository we measure similarity and disimilarities between multiple graphs from topological property level wise and adjacency property level wise. Each method in the `main.py` file gives a vector in a dataframe or a float value. Collecting these results for multile graphs gives a comparative difference between these graphs. 

The similar and disimilar nodes list and edges list between any two graphs can give a primary point of comparison. 

# Analytical data representation

Some node property and edge property information stored into excel file.

## Methods
1. `edgePropertyAnalysis(graphs, True)`: 

    Description: It shows all properties for all the graphs in series. 

    Input: An array of graph file names and a boolean (True indicates storing the output into an excel file and False indicates not storing it anywhere.)

    Output: A dataframe containing all properties as columns and for each nodes as rows. 

2. `nodePropertyAnalysis(graphs, True)`: 

    Description: It shows all properties for all the graphs in series. 

    Input: An array of graph file names and a boolean (True indicates storing the output into an excel file and False indicates not storing it anywhere.)

    Output: A dataframe containing all properties as columns and for each nodes as rows. 

# Topological property level similarity and disimilarity

We considered following node properties to represents the topological structure of a graph:

- [Degree](https://en.wikipedia.org/wiki/Degree_(graph_theory))
- [Degree centrality](https://en.wikipedia.org/wiki/Centrality#Degree_centrality)
- [Closeness centrality](https://en.wikipedia.org/wiki/Centrality#Closeness_centrality) 
- [Betweenness centrality](https://en.wikipedia.org/wiki/Centrality#Betweenness_centrality)
- [Clustering coefficient](https://en.wikipedia.org/wiki/Clustering_coefficient)
- [Eccentricity](https://mathworld.wolfram.com/GraphEccentricity.html) 

## Methods
1. `store_graph(graphs)`:

    Description: This method will store the graphs from the `graphs' array list. 

    Input: To store a graph, we have to put the *graph*.txt file inside the Input_Graphs folder and add the name of the graph file inside the ``graphs`` array.

    Output: A string indicating successful graph storing.

2. `getGraphProperties(graphs)}`: 

    Description: This method will show if the graph is connected, how many number of components are in the graph, total nodes and, total edges of a list of graphs.

    Input: An array of graph file names.

    Output: A dataframe containing the the mentioned basic graph properties.
    

3. `getNodeProperties(graph_name)`:

    Description: This method will calculate all the node properties of the graph.

    Input: The string containing the graph file name.

    Output: A dataframe containing degree, Degree centrality, Closeness centrality, Betweenness centrality, Clustering coefficient, Eccentricity for each node of the graph.

4. `getDifference(graph1, graph2)`:

    Description: It shows the absolute difference between two node property dataframe of two graphs.

    Input: Two strings containing two graph file names.

    Output: A dataframe containing absolute difference of degree, Degree centrality, Closeness centrality, Betweenness centrality, Clustering coefficient, Eccentricity for each node between the graphs.

5. `getStructureValue(graph)`:

    Description: It shows a value representing a row vector in a graph dataframe (where each node of the graph are indexs and each row has degree, Degree centrality, Closeness centrality, Betweenness centrality, Clustering coefficient, Eccentricity as column vectors). It sums up all these values. For any certain row, the maximum of the value can be 6. It is because, all these properties are normalized and each of them will be between 0 to 1. So there are total 6 properties and thus will have maximum value of 6 for each row.

    Input: The string containing the graph file name.

    Output: A dataframe containing a float value for each row of the graph.

6. `similarAnddisimilarNodes(graph1, graph2)`:

    Description: It gives two list of similar and disimilar nodes according to the structure value of node vectors between two graphs.

    Input: Two strings containing two graphs

    Output: Two list of nodes


# Adjacency feature level similarity and disimilarity
The graph files contain edge properties for each edge. The properties are,

- unsigned_weighted_rho
- weighted_rho
- rho

## Methods

1. `getEdgeProperties(graph)`:

    Description: It gives a dataframe containing the properties of each edge.

    Input: The string containing the graph file name.

    Output: A dataframe containing unsigned_weighted_rho, weighted_rho, rho for each edge of the graph.


2. `getEdgePropertyDifferenceDf(graph1, graph2)`:

    Description: It shows the absolute difference between two edge property dataframe of two graphs.

    Input: Two strings containing two graph file names.

    Output: A dataframe containing absolute difference of unsigned_weighted_rho, weighted_rho, rho for each edge between the graphs.

3. `getEdgePropertyValue(graph)`:

    Description: It shows a value representing a row vector in a graph edge property dataframe (where each edge of the graph are indexs and each row has unsigned_weighted_rho, weighted_rho, rho as column vectors). It sums up all these values. For any certain row, the maximum of the value can be 3. It is because, all these properties are normalized and each of them will be between 0 to 1. So there are total 3 properties and thus will have maximum value of 3 for each row.

    Input: The string containing the graph file name.

    Output: A dataframe containing a float value for each row of the graph.

4. `similarAnddisimilarEdges(graph1, graph2)`:

    Description: It gives two list of similar and disimilar edges according to the edge property value of edge vectors between two graphs.

    Input: Two strings containing two graphs

    Output: Two list of edges