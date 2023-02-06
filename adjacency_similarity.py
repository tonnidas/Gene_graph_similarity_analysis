import networkx as nx

from get_node_properties import getNodeProperties
from get_difference import getDifference, getStructureChange


# ----------------------------------------------------------------------------
with open('timepoint0_baseline_graph.txt') as f0: lines_0 = f0.readlines()
with open('timepoint1_induction_graph.txt') as f1: lines_1 = f1.readlines()
with open('timepoint2_necropsy_graph.txt') as f2: lines_2 = f2.readlines()


nodes_timepoint_0, nodes_timepoint_1, nodes_timepoint_2 = [], [], []
edgeList_timepoint_0, edgeList_timepoint_1, edgeList_timepoint_2 = [], [], [] 
edge_dict_0, edge_dict_1, edge_dict_2 = dict(), dict(), dict()


for line in lines_0:
    s = line.split()
    edgeList_timepoint_0.append((s[0], s[1]))
    if s[0] not in nodes_timepoint_0: nodes_timepoint_0.append(s[0])
    if s[1] not in nodes_timepoint_0: nodes_timepoint_0.append(s[1])
    edge_dict_0[(s[0], s[1])] = dict()
    edge_dict_0[(s[0], s[1])] = {'unsigned_weighted_rho': s[3], 'weighted_rho': s[5], 'rho': s[7]}
    # print(s[6])


for line in lines_1:
    s = line.split()
    edgeList_timepoint_1.append((s[0], s[1]))
    if s[0] not in nodes_timepoint_1: nodes_timepoint_1.append(s[0])
    if s[1] not in nodes_timepoint_1: nodes_timepoint_1.append(s[1])
    edge_dict_1[(s[0], s[1])] = dict()
    edge_dict_1[(s[0], s[1])] = {'unsigned_weighted_rho': s[3], 'weighted_rho': s[5], 'rho': s[7]}


for line in lines_2:
    s = line.split()
    edgeList_timepoint_2.append((s[0], s[1]))
    if s[0] not in nodes_timepoint_2: nodes_timepoint_2.append(s[0])
    if s[1] not in nodes_timepoint_2: nodes_timepoint_2.append(s[1])
    edge_dict_2[(s[0], s[1])] = dict()
    edge_dict_2[(s[0], s[1])] = {'unsigned_weighted_rho': s[3], 'weighted_rho': s[5], 'rho': s[7]}


print()
print()


diff_0_1 = 0
e_0_1_directed = 0
e_0_1_undirected = 0
diff_directed_0_1 = 0
diff_undirected_0_1 = 0
for edge in edgeList_timepoint_0:
    a, b = (edge[0], edge[1]), (edge[1], edge[0])
    if a in edgeList_timepoint_0 or b in edgeList_timepoint_0: e_0_1_undirected += 1
    if a in edgeList_timepoint_0: e_0_1_directed += 1
    if a in edgeList_timepoint_1:
        un = abs(float(edge_dict_1[a]['unsigned_weighted_rho'][:-1]) - float(edge_dict_0[edge]['unsigned_weighted_rho'][:-1]))
        wei = abs(float(edge_dict_1[a]['weighted_rho'][:-1]) - float(edge_dict_0[edge]['weighted_rho'][:-1]))
        rho = abs(float(edge_dict_1[a]['rho'][:-1]) - float(edge_dict_0[edge]['rho'][:-1]))
        diff_directed_0_1 = diff_directed_0_1 + (un + wei + rho)
    if b in edgeList_timepoint_1:
        un = abs(float(edge_dict_1[b]['unsigned_weighted_rho'][:-1]) - float(edge_dict_0[edge]['unsigned_weighted_rho'][:-1]))
        wei = abs(float(edge_dict_1[b]['weighted_rho'][:-1]) - float(edge_dict_0[edge]['weighted_rho'][:-1]))
        rho = abs(float(edge_dict_1[b]['rho'][:-1]) - float(edge_dict_0[edge]['rho'][:-1]))
        diff_undirected_0_1 = diff_undirected_0_1 + (un + wei + rho)
print("Number of G0 edges =", len(edgeList_timepoint_0), "Number of G1 edges =", len(edgeList_timepoint_1), "Matched edges (undirected) =", e_0_1_undirected, "Matched edges (directed) =", e_0_1_directed)
print("Total diff as directed =", diff_directed_0_1, "Total diff as undirected =", diff_undirected_0_1)

print()
print()

diff_1_2 = 0
e_1_2_directed = 0
e_1_2_undirected = 0
diff_directed_1_2 = 0
diff_undirected_1_2 = 0
for edge in edgeList_timepoint_1:
    a, b = (edge[0], edge[1]), (edge[1], edge[0])
    if a in edgeList_timepoint_0 or b in edgeList_timepoint_0: e_1_2_undirected += 1
    if a in edgeList_timepoint_0: e_1_2_directed += 1
    if a in edgeList_timepoint_2:
        un = abs(float(edge_dict_2[a]['unsigned_weighted_rho'][:-1]) - float(edge_dict_1[edge]['unsigned_weighted_rho'][:-1]))
        wei = abs(float(edge_dict_2[a]['weighted_rho'][:-1]) - float(edge_dict_1[edge]['weighted_rho'][:-1]))
        rho = abs(float(edge_dict_2[a]['rho'][:-1]) - float(edge_dict_1[edge]['rho'][:-1]))
        diff_directed_1_2 = diff_directed_1_2 + (un + wei + rho)
    if b in edgeList_timepoint_2:
        un = abs(float(edge_dict_2[b]['unsigned_weighted_rho'][:-1]) - float(edge_dict_1[edge]['unsigned_weighted_rho'][:-1]))
        wei = abs(float(edge_dict_2[b]['weighted_rho'][:-1]) - float(edge_dict_1[edge]['weighted_rho'][:-1]))
        rho = abs(float(edge_dict_2[b]['rho'][:-1]) - float(edge_dict_1[edge]['rho'][:-1]))
        diff_undirected_1_2 = diff_undirected_1_2 + (un + wei + rho)
print("Number of G1 edges =", len(edgeList_timepoint_1), "Number of G2 edges =", len(edgeList_timepoint_2), "Matched edges (undirected) =", e_1_2_undirected, "Matched edges (directed) =", e_1_2_directed)
print("Total diff as directed =", diff_directed_1_2, "Total diff as undirected =", diff_undirected_1_2)

print()
print()

diff_2_0 = 0
e_2_0_directed = 0
e_2_0_undirected = 0
diff_directed_2_0 = 0
diff_undirected_2_0 = 0
for edge in edgeList_timepoint_2:
    a, b = (edge[0], edge[1]), (edge[1], edge[0])
    if a in edgeList_timepoint_0 or b in edgeList_timepoint_0: e_2_0_undirected += 1
    if a in edgeList_timepoint_0: e_2_0_directed += 1
    if a in edgeList_timepoint_0:
        un = abs(float(edge_dict_0[a]['unsigned_weighted_rho'][:-1]) - float(edge_dict_2[edge]['unsigned_weighted_rho'][:-1]))
        wei = abs(float(edge_dict_0[a]['weighted_rho'][:-1]) - float(edge_dict_2[edge]['weighted_rho'][:-1]))
        rho = abs(float(edge_dict_0[a]['rho'][:-1]) - float(edge_dict_2[edge]['rho'][:-1]))
        # print(un + wei + rho)
        diff_directed_2_0 = diff_directed_2_0 + (un + wei + rho)
    if b in edgeList_timepoint_0:
        un = abs(float(edge_dict_0[b]['unsigned_weighted_rho'][:-1]) - float(edge_dict_2[edge]['unsigned_weighted_rho'][:-1]))
        wei = abs(float(edge_dict_0[b]['weighted_rho'][:-1]) - float(edge_dict_2[edge]['weighted_rho'][:-1]))
        rho = abs(float(edge_dict_0[b]['rho'][:-1]) - float(edge_dict_2[edge]['rho'][:-1]))
        diff_undirected_2_0 = diff_undirected_2_0 + (un + wei + rho)
print("Number of G2 edges =", len(edgeList_timepoint_2), "Number of G0 edges =", len(edgeList_timepoint_0), "Matched edges (undirected) =", e_2_0_undirected, "Matched edges (directed) =", e_2_0_directed)
print("Total diff as directed =", diff_directed_2_0, "Total diff as undirected =", diff_undirected_2_0)

print()
print()