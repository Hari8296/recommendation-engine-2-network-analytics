Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

G=pd.read_csv("D:/assignments of data science/11 recommendation engine 2 & network analytics/network analytics/flight_hault.csv")

G = G.iloc[0:2000, 1:20] 
G.info()

print(nx.info(G))

g = nx.Graph()
g = nx.from_pandas_edgelist(G, source = 'IATA_FAA', target = 'ICAO')

print(nx.info(g))

d = nx.degree_centrality(g) 
print(d) 

pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, node_size = 30, node_color = 'blue')

closeness = nx.closeness_centrality(g)
print(closeness)

b = nx.betweenness_centrality(g) 
print(b)

cluster_coeff = nx.clustering(g)
print(cluster_coeff)

cc = nx.average_clustering(g) 
print(cc)