# Covid19

Code for SARS-CoV2-Drug Interaction Prediction

Prepare the Adjacencymatrix.csv file.

Run  Node2Vec.py script directly on the above adjacency matrix.

The user input for #Number of Features to be selected using the algorithm. 

Output **Feature_Matrix** is the feature representaion of each node.

Run  Drug-Cov-linkPrediction_Code.py script directly with adjacency matrix and Feature_Matrix.

User should provide testedge.csv  file to predict the edge probability.

## Pre-requisites

> Python version  3.3


> Python Packages: pandas, node2vec, numpy, sklearn, scipy, networkx, tensorflow
