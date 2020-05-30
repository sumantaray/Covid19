# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:26:54 2019

@authors: Snehalika & Sumanta
"""
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_mutual_info_score



'''
Disclaimer: classes from this file come from
tkipf/gae original repository on Graph Autoencoders
'''
# Optimizer Module

class OptimizerVAE(object):
    """ Optimizer for variational autoencoders """
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     labels = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.01) #Learning rate
        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) *                   tf.reduce_mean(tf.reduce_sum(1                                                + 2 * model.z_log_std                                                - tf.square(model.z_mean)                                                - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction =             tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                              tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


#Hidden Layer Module

_LAYER_UIDS = {} # Global unique layer ID dictionary for layer name assignment

"""
Disclaimer: functions and classes from lines 15 to 101 in this file
come from tkipf/gae original repository on Graph Autoencoders.
"""

def get_layer_uid(layer_name = ''):
    """Helper function, assigns unique layer IDs """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """ Graph convolution layer """
    def __init__(self, input_dim, output_dim, adj, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs"""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Symmetric inner product decoder layer"""
    def __init__(self, fastgae, sampled_nodes, dropout = 0., act = tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.sampled_nodes = sampled_nodes # Nodes from sampled subgraph to decode
        self.fastgae = fastgae # Whether to use the FastGAE framework

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        # If FastGAE is used, we only reconstruct the sampled subgraph
        if self.fastgae:
            inputs = tf.gather(inputs, self.sampled_nodes)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

#Weigh Variable Update

def weight_variable_glorot(input_dim, output_dim, name = ""):
    """
    Create a weight variable with Glorot&Bengio (AISTATS 2010) initialization
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval = -init_range,
                                maxval = init_range, dtype = tf.float32)
    return tf.Variable(initial, name = name)


#Model Initialization

class Model(object):
    """ Model base class"""
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelVAE(Model):
    """
    Standard Graph Variational Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder, Gaussian distributions and inner product decoder
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero,hiddendim=32,hiddendim2=16, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.sampled_nodes = placeholders['sampled_nodes']
        self.build()

    def _build(self):
        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = hiddendim, 
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = hiddendim,
                                       output_dim = hiddendim2,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden)

        self.z_log_std = GraphConvolution(input_dim = hiddendim,
                                          output_dim = hiddendim2,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging)(self.hidden)

        self.z = self.z_mean + tf.random_normal([self.n_samples, hiddendim2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(fastgae = True, # Whether to use FastGAE
                                                   sampled_nodes = self.sampled_nodes, # FastGAE subgraph
                                                   act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)


# Preprocessing Module

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def mask_test_edges(adj, test_percent=10., val_percent=10.):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    edges_positive, _, _ = sparse_to_tuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:]

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx] # positive test edges
    val_edges = edges_positive[val_edge_idx] # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0) # positive train edges

    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')

    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not anymore
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()
        # step 4:
        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]
        # step 5:
        coords = np.unique(coords, axis = 0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 7:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis = 0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train,val_edges, val_edges_false, test_edges, test_edges_false


# ROC, Precision Score

def sigmoid(x):
    """ Sigmoid activation function
    :param x: scalar value
    :return: sigmoid activation
    """
    return 1 / (1 + np.exp(-x))

def get_roc_score(edges_pos, edges_neg,emb):
    """ Link Prediction: computes AUC ROC and AP scores from embeddings vectors,
    and from ground-truth lists of positive and negative node pairs
    :param edges_pos: list of positive node pairs
    :param edges_neg: list of negative node pairs
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :return: Area Under ROC Curve (AUC ROC) and Average Precision (AP) scores
    """
   
    #if emb is None:
    #    feed_dict.update({placeholders['dropout']: 0})
    #    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    
    preds = []
    preds_neg = []
    for e in edges_pos:
        # Link Prediction on positive pairs, return preds to get probability
        preds.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
    for e in edges_neg:
        # Link Prediction on negative pairs
        preds_neg.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))

    # Stack all predictions and labels
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    # Computes metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def clustering_latent_space(emb, label, nb_clusters=None):
    """ Node Clustering: computes Adjusted Mutual Information score from a
    K-Means clustering of nodes in latent embedding space
    :param emb: n*d matrix of embedding vectors for all graph nodes
    :param label: ground-truth node labels
    :param nb_clusters: int number of ground-truth communities in graph
    :return: Adjusted Mutual Information (AMI) score
    """
    if nb_clusters is None:
        nb_clusters = len(np.unique(label))
    # K-Means Clustering
    clustering_pred = KMeans(n_clusters = nb_clusters, init = 'k-means++').fit(emb).labels_
    # Compute metrics
    return adjusted_mutual_info_score(label, clustering_pred)


# Compute transition Probability


#wn.simplefilter('ignore', UserWarning)

def get_distribution(measure, alpha, adj):
    """ Compute the p_i probabilities to pick each node i through the
    node sampling scheme of FastGAE (see subsection 3.2.3. of paper)
    :param measure: node importance measure, among 'degree', 'core', 'uniform'
    :param alpha: alpha scalar hyperparameter for degree and core sampling
    :param adj: sparse adjacency matrix of the graph
    :return: list of p_i probabilities of all nodes
    """
    if measure == 'degree':
        # Degree-based distribution
        proba = np.power(np.sum(adj, axis=0),alpha).tolist()[0]
    elif measure == 'core':
        # Core-based distribution
        G = nx.from_scipy_sparse_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        proba = np.power(list(nx.core_number(G).values()), alpha)
    elif measure == 'uniform':
        # Uniform distribution
        proba = np.ones(adj.shape[0])
    else:
        raise ValueError('Undefined sampling method!')
    # Normalization
    proba = proba/np.sum(proba)
    return proba

def node_sampling(adj, distribution, nb_node_samples, replace=False):
    """ Sample a subgraph from a given node-level distribution
    :param adj: sparse adjacency matrix of the graph
    :param distribution: p_i distribution, from get_distribution()
    :param nb_node_samples: size (nb of nodes) of the sampled subgraph
    :param replace: whether to sample nodes with replacement
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    # Sample nb_node_samples nodes, from the pre-computed distribution
    sampled_nodes = np.random.choice(adj.shape[0], size = nb_node_samples,
                                     replace = replace, p = distribution)
    # Sparse adjacency matrix of sampled subgraph
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    # In tuple format (useful for optimizers)
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))
    return sampled_nodes, sampled_adj_tuple, sampled_adj


# Load Input Data

#Input data loading
mat = pd.read_csv('/export/scratch2/sumanta/adjacencymatrix.csv', index_col=0)
feature_mat = pd.read_csv('/export/scratch2/sumanta/feature_matrix.csv',header=-1) # index_col=0


# Traning Model

epochs = 20  # Number of epoch, User Defined
snode=1000   # Number of Sampling Nodes, Used Defined
mean_roc = []
mean_ap = []
for i in range(epochs):
    print('epochs:',i)
    net_numpy=mat.to_numpy()
    G = nx.from_numpy_matrix(net_numpy)
    adj_init = nx.adjacency_matrix(G)
    features_init = sp.lil_matrix(feature_mat)
 
    np.random.seed(0) # IMPORTANT: guarantees consistent train/test splits
    adj, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_init,10.,10)
    
    num_nodes = adj.shape[0]
    features = features_init
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
         
    import time
    t_start = time.time()
    node_distribution = get_distribution('degree', 1.0, adj)  # measure=degree, alpha=1.0(Sharpening Parameter)
    sampled_nodes, adj_label, adj_sampled_sparse = node_sampling(adj, node_distribution,snode, False) #Sampling node,Replace False
    hiddendim = 32          #Number of Hidden Layer1
    hiddendim2 = 16         #Number of Hidden Layer2
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ()),
        'sampled_nodes': tf.placeholder_with_default(sampled_nodes, shape = [snode])}
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,hiddendim, hiddendim2)
    num_sampled = adj_sampled_sparse.shape[0]
    sum_sampled = adj_sampled_sparse.sum()
    pos_weight = float(num_sampled * num_sampled - sum_sampled) / sum_sampled
    norm = num_sampled * num_sampled / float((num_sampled * num_sampled
                                                    - sum_sampled) * 2)
    opt = OptimizerVAE(preds = model.reconstructions,
                           labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                         validate_indices = False), [-1]),
                           model = model,
                           num_nodes = num_nodes,
                           pos_weight = pos_weight,
                           norm = norm)
    adj_norm = preprocess_graph(adj)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

 # Model training
    print("Training...")

    for iter in range(100):    #Training Iteration
        # Flag to compute running time for each iteration
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)

        
        feed_dict.update({placeholders['sampled_nodes']: sampled_nodes})
        sampled_nodes, adj_label, _ = node_sampling(adj, node_distribution,snode, False) #Sampling Node

        # Weights update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict = feed_dict)
        # Compute average loss and acc
        
        avg_cost = outs[1]
        # Print Number of Iter, Training loss, Time
        print("Iter:", '%04d' % (iter + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
 
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict = feed_dict)
        feed_dict.update({placeholders['dropout']: 0})
        val_roc, val_ap = get_roc_score(val_edges, val_edges_false,emb)
        
        #Print Validation ROC and Average Precision score
        print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))


    # Test model
    print("Testing model...")
    emb = sess.run(model.z_mean, feed_dict = feed_dict)   
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false,emb)
    print("test_roc=", "{:.5f}".format(roc_score), "test_ap=", "{:.5f}".format(ap_score))
        # Report scores
    mean_roc.append(roc_score)
    mean_ap.append(ap_score)
   


# Print Testing results


print("\nTest results for", 'gcn_vae',
      "model on", 'Drug_Prediction', "on", 'Link_Prediction', "\n",
      "___________________________________________________\n")

#if FLAGS.task == 'link_prediction':
print("AUC scores\n", mean_roc)
print("Mean AUC score: ", np.mean(mean_roc),
          "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")

print("AP scores\n", mean_ap)
print("Mean AP score: ", np.mean(mean_ap),
          "\nStd of AP scores: ", np.std(mean_ap), "\n \n")


# Load the test edges

testedges = pd.read_csv('testedges.csv',index_col=0) # index_col=0
testpred = testedges.to_numpy()
testpred=testpred-1

#Predict the test edge probability
preds = []
testpred=testpred
for e in testpred:
    preds.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
    


# Save the probability of predicted edges


np.savetxt('predprob.csv', preds, delimiter=',')





