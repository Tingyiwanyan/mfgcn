import os
import json
import numpy as np
import random
import networkx as nx
from data_load import Data_loading

class Data_process(Data_loading):
    """
    Data process, cut edges, nodes, create transductive and inductive training ,testing set
    """
    def __init__(self,data_set,option_lp_nc):
        Data_loading.__init__(self,data_set)
        self.dic_non_edges = {}
        self.non_edges = None
        self.edges = list(self.G.edges())
        self.nodes = list(self.G.nodes())
        self.n_edges = self.G.number_of_edges()
        self.n_nodes = self.G.number_of_nodes()
        self.n_non_edges = None
        self.G_total = self.G.copy()
        #self.train_inductive_G = self.G.copy()
        self.rnd = np.random.RandomState(seed=None)
        self.train_cut_edge = None
        self.prop_pos = 0.5
        self.prop_neg = 0.5
        self.prop_nc = 0.1
        self.npos = np.int(self.prop_pos*self.n_edges)
        self.n_nc = np.int(self.prop_nc*self.n_nodes)
        self.nneg = None
        self.neg_edge_test = None
        self.train_nodes = None
        self.test_nodes = None
        self.option_lp_nc = option_lp_nc

        #self.non_edges_dic()

    def non_edges_dic(self):
        self.non_edges = [e for e in nx.non_edges(self.G_total)]
        self.nneg = np.int(self.prop_neg*len(self.non_edges))
        rnd_inx_neg = self.rnd.choice(len(self.non_edges), self.nneg, replace=False)
        self.neg_edge_test = [self.non_edges[i] for i in rnd_inx_neg]
        for i in range(len(self.non_edges)):
            self.dic_non_edges.setdefault(self.non_edges[i][0],[]).append(self.non_edges[i][1])

    def generate_train_graph(self):
        rnd_inx = self.rnd.choice(len(self.edges), self.npos, replace=False)
        self.train_cut_edges = [self.edges[i] for i in rnd_inx]
        for edge in self.train_cut_edges:
            if self.G.has_edge(edge[0],edge[1]):
                self.G.remove_edge(edge[0],edge[1])
            if self.G.has_edge(edge[1],edge[0]):
                self.G.remove_edge(edge[1],edge[0])
            if len(list(self.G.neighbors(edge[0]))) == 0:
                self.G.add_edge(edge[0],edge[1])
                self.G.add_edge(edge[1],edge[0])
                self.train_cut_edges.remove((edge[0],edge[1]))

    def generate_train_node(self):
        rnd_index = self.rnd.choice(len(self.nodes), self.n_nc, replace=False)
        #rnd_index = range(self.n_nc)
        self.train_nodes = [self.nodes[i] for i in rnd_index]
        self.test_nodes = [x for x in self.nodes if x not in self.train_nodes]

    #def generate_inductive_train_graph(self):

    def config_train_test(self):
        if self.option_lp_nc == 1:
            """
            generate train graph for transductive link prediction
            """
            self.non_edges_dic()
            self.generate_train_graph()
        if self.option_lp_nc == 2:
            """
            generate train nodes for transductive node classification
            """
            self.non_edges_dic()
            self.generate_train_node()







   # def random_cut_edges(self):
      #  g_cut = self.G.copy()
      #  g_edges =





