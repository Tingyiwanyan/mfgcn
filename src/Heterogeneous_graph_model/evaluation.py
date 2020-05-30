import os
import json
import numpy as np
import random
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from utils import utils
import matplotlib.pyplot as plt

class evaluation(object):
    def __init__(self,utils,option_lp_nc):
        if option_lp_nc == 3:
            """
            change test nodes into whole graph
            """
            self.G = utils.G_test
            utils.G = utils.G_test
        else:
            self.G = utils.G
        self.data_length = len(list(self.G.nodes()))
        self.attribute_size = utils.attribute_size
        self.pos_test_edges = None
        self.neg_test_edges = utils.neg_edge_test
        self.batch_size = 1
        self.walk_length = utils.walk_length
        self.negative_sample_size = utils.negative_sample_size
        self.score_pos = None
        self.score_neg = None
        self.test_number_neg = 5000
        self.roc_resolution = 0.1
        self.test_number_pos = None
        self.train_nodes = utils.train_nodes
        self.latent_dim = utils.latent_dim
        self.test_nodes = utils.test_nodes
        if option_lp_nc == 1:
            self.pos_test_edges = utils.train_cut_edges
            #self.test_number_pos = len(self.pos_test_edges)
            self.test_number_pos = 3000


    def get_test_embed_mfgcn(self,node,utils):
        mini_batch_integral = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.attribute_size))
        batch_GCN_agg = utils.get_batch_GCNagg([node])
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

    def get_test_embed_raw(self,node,utils):
        mini_batch_integral_center = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.attribute_size))
        batch_x_center = utils.get_minibatch([node])
        mini_batch_integral_center[0, 0, :] = batch_x_center[0, :]
        return mini_batch_integral_center


    def get_test_embed_meanpool(self,node,utils):
        mini_batch_integral_meanpool = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size, utils.attribute_size))
        # batch_center_x = self.get_minibatch(node)
        batch_meanpool = utils.get_batch_mean_pooling_neighbor([node])
        # for i in range(self.batch_size):
        mini_batch_integral_meanpool[0, 0, :] = batch_meanpool[0, :]
        return mini_batch_integral_meanpool

    def get_test_embed_maxpool(self,node,utils):
        mini_batch_integral_maxpool = np.zeros(
            (1, 1 + utils.walk_length + utils.negative_sample_size,utils.neighborhood_sample_num, utils.attribute_size))
        batch_maxpool = utils.get_batch_max_pooling([node])
        mini_batch_integral_maxpool[0,0,:,:] = batch_maxpool[0,:,:]

        return mini_batch_integral_maxpool



    def evaluate_lp(self,utils):
        self.score_pos = np.zeros(self.test_number_pos)
        #self.score_pos = np.zeros(self.test_number)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == self.test_number_pos - 1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0],utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1],utils)
            x_center1 = self.get_test_embed_raw(test_sample[0],utils)
            x_center2 = self.get_test_embed_raw(test_sample[1],utils)
            embed1 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn1,utils.x_center:x_center1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn2,utils.x_center:x_center2})[0][0,0,:]
            embed1_norm = embed1/np.linalg.norm(embed1)
            embed2_norm = embed2/np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1
            #print(i)

        i = 0
        #self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(self.test_number_neg)
        for test_sample in self.neg_test_edges:
            if i == self.test_number_neg - 1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0],utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1],utils)
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            embed1 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn1,utils.x_center:x_center1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn2,utils.x_center:x_center2})[0][0,0,:]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1
            #print(i)

    def evaluate_n2v_lp(self,utils):
        self.score_pos = np.zeros(self.test_number_pos)
        #self.score_pos = np.zeros(self.test_number)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == self.test_number_pos - 1:
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
        self.score_neg = np.zeros(self.test_number_neg)
        for test_sample in self.neg_test_edges:
            if i == self.test_number_neg - 1:
                break
            x_n2v1 = self.get_test_embed_n2v(test_sample[0],utils)
            x_n2v2 = self.get_test_embed_n2v(test_sample[1],utils)
            embed1 = utils.sess.run([utils.Dense4_n2v], feed_dict={utils.x_n2v: x_n2v1})[0][0,0,:]
            embed2 = utils.sess.run([utils.Dense4_n2v], feed_dict={utils.x_n2v: x_n2v2})[0][0,0,:]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm,embed2_norm))
            i = i+1

    def evaluate_meanpool_lp(self,utils):
        self.score_pos = np.zeros(self.test_number_pos)
        # self.score_pos = np.zeros(self.test_number)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == self.test_number_pos - 1:
                break
            x_meanpool1 = self.get_test_embed_meanpool(test_sample[0], utils)
            x_meanpool2 = self.get_test_embed_meanpool(test_sample[1], utils)
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            embed1 = \
            utils.sess.run([utils.Dense_mean_pool_final], feed_dict={utils.x_mean_pool:x_meanpool1,utils.x_center:x_center1})[0][0,0,:]
            embed2 = \
            utils.sess.run([utils.Dense_mean_pool_final], feed_dict={utils.x_mean_pool:x_meanpool2,utils.x_center:x_center2})[0][0,0,:]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1

        i = 0
        # self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(self.test_number_neg)
        for test_sample in self.neg_test_edges:
            if i == self.test_number_neg - 1:
                break
            x_meanpool1 = self.get_test_embed_meanpool(test_sample[0], utils)
            x_meanpool2 = self.get_test_embed_meanpool(test_sample[1], utils)
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            embed1 = \
                utils.sess.run([utils.Dense_mean_pool_final],
                               feed_dict={utils.x_mean_pool: x_meanpool1, utils.x_center: x_center1})[0][0, 0, :]
            embed2 = \
                utils.sess.run([utils.Dense_mean_pool_final],
                               feed_dict={utils.x_mean_pool: x_meanpool2, utils.x_center: x_center2})[0][0, 0, :]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1

    def evaluate_maxpool_lp(self,utils):
        self.score_pos = np.zeros(self.test_number_pos)
        # self.score_pos = np.zeros(self.test_number)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == self.test_number_pos - 1:
                break
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            x_maxpool1 = self.get_test_embed_maxpool(test_sample[0], utils)
            x_maxpool2 = self.get_test_embed_maxpool(test_sample[1], utils)
            embed1 = utils.sess.run([utils.Dense_layer_maxpool], feed_dict={utils.x_center: x_center1,
                                                                            utils.x_max_pool: x_maxpool1})
            embed2 = utils.sess.run([utils.Dense_layer_maxpool], feed_dict={utils.x_center: x_center2,
                                                                             utils.x_max_pool: x_maxpool2})
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1

        i = 0
        # self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(self.test_number_neg)
        for test_sample in self.neg_test_edges:
            if i == self.test_number_neg - 1:
                break
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            x_maxpool1 = self.get_test_embed_maxpool(test_sample[0], utils)
            x_maxpool2 = self.get_test_embed_maxpool(test_sample[1], utils)
            embed1 = utils.sess.run([utils.Dense_layer_maxpool], feed_dict={utils.x_center: x_center1,
                                                                            utils.x_max_pool: x_maxpool1})
            embed2 = utils.sess.run([utils.Dense_layer_maxpool], feed_dict={utils.x_center: x_center2,
                                                                            utils.x_max_pool: x_maxpool2})
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1

    def evaluate_raw_lp(self,utils):
        self.score_pos = np.zeros(self.test_number_pos)
        # self.score_pos = np.zeros(self.test_number)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == self.test_number_pos - 1:
                break
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            embed1 = \
            utils.sess.run([utils.Dense_raw_final], feed_dict={utils.x_center:x_center1})[0][0,0,:]
            embed2 = \
            utils.sess.run([utils.Dense_raw_final], feed_dict={utils.x_center:x_center2})[0][0,0,:]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_pos[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1

        i = 0
        # self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(self.test_number_neg)
        for test_sample in self.neg_test_edges:
            if i == self.test_number_neg - 1:
                break
            x_center1 = self.get_test_embed_raw(test_sample[0], utils)
            x_center2 = self.get_test_embed_raw(test_sample[1], utils)
            embed1 = \
                utils.sess.run([utils.Dense_raw_final],
                               feed_dict={ utils.x_center: x_center1})[0][0, 0, :]
            embed2 = \
                utils.sess.run([utils.Dense_raw_final],
                               feed_dict={utils.x_center: x_center2})[0][0, 0, :]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1


    def evaluate_combined_lp(self,utils):
        self.score_pos = np.zeros(self.test_number_pos)
        #self.score_pos = np.zeros(self.test_number)
        i = 0
        for test_sample in self.pos_test_edges:
            if i == self.test_number_pos - 1:
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
            #print(i)

        i = 0
        # self.score_neg = np.zeros(len(self.neg_test_edges))
        self.score_neg = np.zeros(self.test_number_neg)
        for test_sample in self.neg_test_edges:
            if i == self.test_number_neg - 1:
                break
            x_gcn1 = self.get_test_embed_mfgcn(test_sample[0], utils)
            x_gcn2 = self.get_test_embed_mfgcn(test_sample[1], utils)
            x_n2v1 = self.get_test_embed_n2v(test_sample[0], utils)
            x_n2v2 = self.get_test_embed_n2v(test_sample[1], utils)
            embed1 = utils.sess.run([utils.x_origin], feed_dict={utils.x_n2v: x_n2v1, utils.x_gcn: x_gcn1})[0][0, 0, :]
            embed2 = utils.sess.run([utils.x_origin], feed_dict={utils.x_n2v: x_n2v2, utils.x_gcn: x_gcn2})[0][0, 0, :]
            embed1_norm = embed1 / np.linalg.norm(embed1)
            embed2_norm = embed2 / np.linalg.norm(embed2)
            self.score_neg[i] = np.sum(np.multiply(embed1_norm, embed2_norm))
            i = i + 1
            #print(i)
    def evaluate_combine_nc(self,utils):
        predict_correct = 0.0
        for test_sample in self.test_nodes:
            x_n2v = self.get_test_embed_n2v(test_sample,utils)
            x = self.get_test_embed_mfgcn(test_sample,utils)
            x_center = self.get_test_embed_raw(test_sample,utils)
            logit = utils.sess.run([utils.logit_softmax_reduce],feed_dict={utils.x_n2v:x_n2v,utils.x_gcn:x,
                                                                           utils.x_center:x_center})
            predict = np.where(logit[0] == logit[0].max())[0][0]
            if predict == utils.G.nodes[test_sample]['label']:
                predict_correct += 1
        self.tp_rate = predict_correct/np.float(len(self.test_nodes))

    def evaluate_n2v_nc(self,utils):
        predict_correct = 0.0
        for test_sample in self.test_nodes:
            x_n2v = self.get_test_embed_n2v(test_sample,utils)
            #x = self.get_test_embed_mfgcn(test_sample,utils)
            logit = utils.sess.run([utils.logit_softmax_reduce],feed_dict={utils.x_n2v:x_n2v})
            predict = np.where(logit[0] == logit[0].max())[0][0]
            if predict == utils.G.nodes[test_sample]['label']:
                predict_correct += 1
        self.tp_rate = predict_correct/np.float(len(self.test_nodes))

    def evaluate_combine_meanpool_nc(self,utils):
        predict_correct = 0.0
        for test_sample in self.test_nodes:
            x_n2v = self.get_test_embed_n2v(test_sample, utils)
            x = self.get_test_embed_meanpool(test_sample,utils)
            x_center = self.get_test_embed_raw(test_sample,utils)
            logit = utils.sess.run([utils.logit_softmax_reduce], feed_dict={utils.x_mean_pool:x,utils.x_n2v:x_n2v,
                                                                            utils.x_center:x_center})
            predict = np.where(logit[0] == logit[0].max())[0][0]
            if predict == utils.G.nodes[test_sample]['label']:
                predict_correct += 1
        self.tp_rate = predict_correct / np.float(len(self.test_nodes))

    def evaluate_combine_raw_nc(self,utils):
        predict_correct = 0.0
        for test_sample in self.test_nodes:
            x_n2v = self.get_test_embed_n2v(test_sample, utils)
            x = self.get_test_embed_raw(test_sample,utils)
            logit = utils.sess.run([utils.logit_softmax_reduce], feed_dict={utils.x_center:x,utils.x_n2v:x_n2v})
            predict = np.where(logit[0] == logit[0].max())[0][0]
            if predict == utils.G.nodes[test_sample]['label']:
                predict_correct += 1
        self.tp_rate = predict_correct / np.float(len(self.test_nodes))

    def evaluate_max_pool_nc(self,utils):
        predict_correct = 0.0
        for test_sample in self.test_nodes:
            x_n2v = self.get_test_embed_n2v(test_sample, utils)
            x = self.get_test_embed_raw(test_sample, utils)
            x_maxpool = self.get_test_embed_maxpool(test_sample,utils)
            logit = utils.sess.run([utils.logit_softmax_reduce], feed_dict={utils.x_center: x, utils.x_n2v: x_n2v,
                                                                            utils.x_max_pool:x_maxpool})
            predict = np.where(logit[0] == logit[0].max())[0][0]
            if predict == utils.G.nodes[test_sample]['label']:
                predict_correct += 1
        self.tp_rate = predict_correct / np.float(len(self.test_nodes))

    """
    generate kmeans, only for mf-gcn
    """
    def generate_kmeans(self,utils):
        nodes_total = [i for i in self.G.nodes]
        len_nodes = len(self.G.nodes())
        final_embed = np.zeros((len_nodes,self.latent_dim))
        embed_label = np.zeros((len_nodes))
        index = 0
        for i in nodes_total:
            embed_label[index] = i
            x_gcn1 = self.get_test_embed_mfgcn(i, utils)
            x_center1 = self.get_test_embed_raw(i, utils)
            embed1 = \
            utils.sess.run([utils.Dense_layer_fc_gcn], feed_dict={utils.x_gcn: x_gcn1, utils.x_center: x_center1})[0][0,
            0, :]
            final_embed[index,:] = embed1
            index += 1

        return final_embed,embed_label



def generate_kmeans_single_filter(utils,evl,filter_num):
    len_nodes = len(utils.G.nodes())
    embed_filter = np.zeros((len_nodes,16))
    embed_label = np.zeros((len_nodes))
    index = 0
    for i in utils.G.nodes():
        embed_label[index] = i
        x_gcn1 = evl.get_test_embed_mfgcn(i,utils)
        x_center1 = evl.get_test_embed_mfgcn(i,utils)
        embed1 = \
            utils.sess.run(utils.Dense_gcn_layers,
                           feed_dict={utils.x_gcn: x_gcn1,
                                      utils.x_center: x_center1,utils.train_test_mode:False})[filter_num][0,0, :]
        embed_filter[index,:] = embed1
        index += 1
    kmeans = KMeans(n_clusters=2,random_state=0).fit(embed_filter)

    return embed_filter,embed_label,kmeans.labels_





def cal_auc(score_pos,score_neg,roc_resolution,test_number_pos,test_number_neg):
    threshold = -1
    tp_rates = []
    fp_rates = []

    while(threshold < 1.001):
        tpr = len(np.where(score_pos>threshold)[0])/np.float(test_number_pos)
        fpr = len(np.where(score_neg>threshold)[0])/np.float(test_number_neg)
        tp_rates.append(tpr)
        fp_rates.append(fpr)
        threshold += roc_resolution
    area = 0
    tp_rates.reverse()
    fp_rates.reverse()
    for i in range(len(tp_rates)-1):
        x = fp_rates[i+1]-fp_rates[i]
        y = (tp_rates[i+1]+tp_rates[i])/2
        area += x*y
    return tp_rates,fp_rates,area

class Plot_roc(object):
    def __init__(self):
        self.setup_roc()

    def setup_roc(self):
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve", fontsize=14)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')
    def add_roc_curve(self, tp_rates,fp_rates,color,label):
        #tp_rates = np.array(tp_rates)
        #fp_rates = np.array(fp_rates)
        plt.plot(fp_rates,tp_rates,color=color,linewidth=1, label=label)
    def show_plot(self):
        plt.legend(loc='lower right')
        plt.show()

def write_file(rates,name):
    file = open(name,'w')
    for element in rates:
        print>>file, element
    file.close

def read_file(name):
    file = open(name)
    rates = []
    for line in file:
        line = line.rstrip('\n')
        rates.append(line)
    rates = [np.float(i) for i in rates]
    rates = np.array(rates)
    return rates

def write_file_kmeans(file_name,kmeans_label,embed_label):
    #with open(file_name_tp,"w") as output:
       # output.write(str(self.tp_rate_roc))
    #with open(file_name_fp,"w") as output:
        #output.write(str(self.fp_rate_roc))
    with open(file_name,'w') as file:
        for i in range(5638):
            file.write(str(np.int(embed_label[i])))
            file.write(' ')
            file.write(str(kmeans_label[i]))
            file.write('\n')



