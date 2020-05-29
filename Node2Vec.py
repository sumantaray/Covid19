# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:28:03 2020

@author: Sumanta Ray, Snehalika Lall
"""

import pandas as pd
import ast
import pickle
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
from node2vec import Node2Vec
#from multiprocessing import Pool
mat = pd.read_csv('adjacencymatrix.csv', index_col=0) # Upload the adjacency matrix 
net_numpy=mat.to_numpy()
G = nx.from_numpy_matrix(net_numpy)
node2vec = Node2Vec(G, dimensions=100, walk_length=16, num_walks=30,workers=1)
model = node2vec.fit(window=10, min_count=1)

Feature_mat=np.empty(('node','dim'))  #put node = number of nodes, dim= number of required in feature representation
feature_mat[:] = np.nan
for p in range('node'):
    feature_mat[p,:]=model.wv[str(p)]

np.savetxt('feature_matrix.csv', feature_mat, delimiter=',')  