import os
import json
import numpy as np
import random
import networkx as nx
from utils import utils

class evaluation(object):
    def __init__(self,utils):
        self.G = utils.G
        self.data_length = len(list(self.G.nodes()))
        self.attribute_size = utils.attribute_size
        self.pos_test_edges = utils.train_cut_edges
        self.neg_test_edges = utils.neg_edge_test
        self.batch_size = 1
        self.walk_length = utils.walk_length
        self.negative_sample_size = utils.negative_sample_size
        self.score_pos = None
        self.score_neg = None

    def get_test_embed_mfgcn(self,node,utils):
        mini_batch_integral = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.attribute_size))
        #batch_center_x = self.get_minibatch(node)
        batch_GCN_agg = utils.get_batch_GCNagg([node])
        #for i in range(self.batch_size):
        mini_batch_integral[0, 0, :] = batch_GCN_agg[0, :]
        return mini_batch_integral

    def get_test_embed_n2v(self,node,utils):
        mini_batch_integral_n2v = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.length))
        #batch_center_x = self.get_minibatch(node)
        batch_n2v = utils.get_batch_n2v([node])
        #for i in range(self.batch_size):
        mini_batch_integral_n2v[0, 0, :] = batch_n2v[0, :]
        return mini_batch_integral_n2v

    def get_test_embed_combine(self,node,utils):
        mini_batch_integral = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.attribute_size))
        mini_batch_integral_n2v = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.length))
        batch_GCN_agg = utils.get_batch_GCNagg([node])
        # for i in range(self.batch_size):
        mini_batch_integral[0, 0, :] = batch_GCN_agg[0, :]
        batch_n2v = utils.get_batch_n2v([node])
        # for i in range(self.batch_size):
        mini_batch_integral_n2v[0, 0, :] = batch_n2v[0, :]

        return mini_batch_integral,mini_batch_integral_n2v


    def evaluate(self,utils):
        #self.score_pos = np.zeros(len(self.pos_test_edges))
        self.score_pos = np.zeros(7000)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == 7000-1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0],utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1],utils)
            embed1 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn2})[0][0,0,:]
            embed1_norm = embed1/np.linalg.norm(embed1)
            embed2_norm = embed2/np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1

        i = 0
        #self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(7000)
        for test_sample in self.neg_test_edges:
            if i == 7000-1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0],utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1],utils)
            embed1 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn2})[0][0,0,:]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1

    def evaluate_n2v(self,utils):
        #self.score_pos = np.zeros(len(self.pos_test_edges))
        self.score_pos = np.zeros(7000)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == 7000-1:
                break
            x_n2v1 = self.get_test_embed_n2v(test_sample[0],utils)
            x_n2v2 = self.get_test_embed_n2v(test_sample[1],utils)
            embed1 = utils.sess.run([utils.Dense4_n2v], feed_dict={utils.x_n2v: x_n2v1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense4_n2v], feed_dict={utils.x_n2v: x_n2v2})[0][0,0,:]
            embed1_norm = embed1/np.linalg.norm(embed1)
            embed2_norm = embed2/np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1

        i = 0
        #self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(7000)
        for test_sample in self.neg_test_edges:
            if i == 7000-1:
                break
            x_n2v1 = self.get_test_embed_n2v(test_sample[0],utils)
            x_n2v2 = self.get_test_embed_n2v(test_sample[1],utils)
            embed1 = utils.sess.run([utils.Dense4_n2v], feed_dict={utils.x_n2v: x_n2v1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense4_n2v], feed_dict={utils.x_n2v: x_n2v2})[0][0,0,:]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1

    def evaluate_combined(self,utils):
        self.score_pos = np.zeros(7000)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == 7000 - 1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0], utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1], utils)
            x_n2v1 = self.get_test_embed_n2v(test_sample[0], utils)
            x_n2v2 = self.get_test_embed_n2v(test_sample[1], utils)
            embed1 = utils.sess.run([utils.x_origin], feed_dict={utils.x_n2v: x_n2v1,utils.x_gcn:x_gcn1})[0][0, 0, :]
            embed2 = utils.sess.run([utils.x_origin], feed_dict={utils.x_n2v: x_n2v2,utils.x_gcn:x_gcn2})[0][0, 0, :]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1
            print(i)

        i = 0
        # self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(7000)
        for test_sample in self.neg_test_edges:
            if i == 7000 - 1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0], utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1], utils)
            x_n2v1 = self.get_test_embed_n2v(test_sample[0], utils)
            x_n2v2 = self.get_test_embed_n2v(test_sample[1], utils)
            embed1 = utils.sess.run([utils.x_origin], feed_dict={utils.x_n2v: x_n2v1,utils.x_gcn:x_gcn1})[0][0, 0, :]
            embed2 = utils.sess.run([utils.x_origin], feed_dict={utils.x_n2v: x_n2v2,utils.x_gcn:x_gcn2})[0][0, 0, :]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1
            print(i)



