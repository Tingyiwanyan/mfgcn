import tensorflow as tf
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import networkx as nx

class visualization(object):
    """
    2d visualization
    """
    def __init__(self,utils,evaluation):
        self.evl = evaluation
        self.utils = utils
        self.label_whole = np.zeros((utils.G.number_of_nodes()))
        self.test_nodes = utils.test_nodes
        self.embedding_whole = np.zeros((len(self.test_nodes), utils.latent_dim))

    def get_2d_rep(self):
        i = 0
        for node in self.test_nodes:
            x_n2v= self.evl.get_test_embed_n2v(node,self.utils)
            x_gcn = self.evl.get_test_embed_mfgcn(node, self.utils)
            x_center = self.evl.get_test_embed_raw(node, self.utils)
            x_mean = self.evl.get_test_embed_meanpool(node, self.utils)
            x_maxpool = self.evl.get_test_embed_maxpool(node, self.utils)
            embed = self.utils.sess.run([self.utils.x_origin], feed_dict={self.utils.x_n2v: x_n2v, self.utils.x_gcn: x_gcn,
                                                                self.utils.x_center: x_center,self.utils.x_mean_pool:x_mean,
                                                                self.utils.x_max_pool:x_maxpool})[0][0,0,:]
            self.embedding_whole[i,:] = embed
            self.label_whole[i] = self.utils.G.nodes[node]['label']
            i = i + 1
        self.embedding_2d = TSNE(n_components=2).fit_transform(self.embedding_whole)

    def plot_2d(self):
        for i in range(len(self.test_nodes)):
            if self.label_whole[i] == 0:
                color_ = 'green'
            if self.label_whole[i] == 1:
                color_ = 'blue'
            if self.label_whole[i] == 2:
                color_ = 'red'
            if self.label_whole[i] == 3:
                color_ = 'yellow'
            if self.label_whole[i] == 4:
                color_ = 'black'
            if self.label_whole[i] == 5:
                color_ = 'purple'
            if self.label_whole[i] == 6:
                color_ = 'orange'
            if self.label_whole[i] == 7:
                color_ = 'pink'
            plt.plot(self.embedding_2d[i][0], self.embedding_2d[i][1], '.', color=color_, markersize=3)

        plt.show()
