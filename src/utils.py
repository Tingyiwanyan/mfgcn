import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
from model_optimization import model_optimization
from walk import n2v_walk


class utils(model_optimization):
    def __init__(self,data_set,option,option_lp_nc,option_walk,option_structure):
        model_optimization.__init__(self,data_set,option,option_lp_nc,option_structure)
        self.option_walk = option_walk

    def init_walk_prob(self):
        self.n2v_walk = n2v_walk(self.G, 1, 0.5, self.walk_length)
        if self.option_walk == 1:
            """
            BFS strategy
            """
            self.walk_ = self.BFS_search
        if self.option_walk == 2:
            """
            n2v strategy
            """
            self.walk_ = self.n2v_walk.node2vec_walk
            self.n2v_walk.preprocess_transition_probs()

    def assign_value(self, node_index):
        if self.data_set == 1:
            """
            Aminer dataset
            """
            attribute_vector = np.zeros(5)
            attribute_vector[0] = (np.float(self.G.node[node_index]['country_diversity']) - self.mean_count) / self.std_count
            attribute_vector[1] = (np.float(self.G.node[node_index]['topic_diversity']) - self.mean_top) / self.std_top
            attribute_vector[2] = (np.float(self.G.node[node_index]['productivity_diversity']) - self.mean_prod) / self.std_prod
            attribute_vector[3] = (np.float(self.G.node[node_index]['impact_diversity']) - self.mean_impact) / self.std_impact
            attribute_vector[4] = (np.float(self.G.node[node_index]['scientific_age_diversity']) - self.mean_sci) / self.std_sci
        if self.data_set == 2 or self.data_set == 3 or self.data_set == 4:
            """
            Citeceer dataset
            """
            attribute_vector = np.array(self.G.node[node_index]['feature'])

        return attribute_vector

    def assign_value_n2v(self, node):
        one_sample = np.zeros(self.length)
        node = np.int(node)
        for j in self.G.neighbors(node):
            indexy = self.G.node[j]['node_index']
            one_sample[indexy] = 1

        return one_sample


    """
    mean_pooling skip_gram
    """


    def mean_pooling(self, skip_gram_vector):
        attribute_vector_total = np.zeros(self.attribute_size)
        for index in skip_gram_vector:
            attribute_vector_total += self.assign_value(index)

        return attribute_vector_total / self.walk_length


    """
    Get Neighborhood data
    """


    def get_neighborhood_data(self, node_index):
        neighbors = []
        for g in self.G.neighbors(node_index):
            neighbors.append(np.int(g))

        return neighbors, len(neighbors)


    """
    Get neighborhood from new data
    """

    def find_neighbor(self, node_index):
        author_id = self.G.nodes[node_index]['author_id']
        author_id2 = self.G.nodes[node_index]['author_id2']
        if author_id2 == 0:
            author_id2_v = -1
        else:
            author_id2_v = author_id2
        author_id3 = self.G.nodes[node_index]['author_id3']
        if author_id3 == 0:
            author_id3_v = -1
        else:
            author_id3_v = author_id3
        neighbor = [x for x, y in self.G.nodes(data=True) if
                    y['author_id'] == author_id or y['author_id'] == author_id2_v or y['author_id'] == author_id3_v or
                    y['author_id2'] == author_id or y['author_id2'] == author_id2_v or y['author_id2'] == author_id3_v or
                    y['author_id3'] == author_id or y['author_id3'] == author_id2_v or y['author_id3'] == author_id3_v]
        size = len(neighbor)
        return neighbor, size

    """
    BFS search for nodes
    """


    def BFS_search(self, start_node):
        walk_ = []
        visited = [start_node]
        BFS_queue = [start_node]
       # neighborhood = get_neighborhood_data
        while len(walk_) < self.walk_length:
            cur = np.int(BFS_queue.pop(0))
            walk_.append(cur)
            cur_nbrs = sorted(self.get_neighborhood_data(cur)[0])
            for node_bfs in cur_nbrs:
                if not node_bfs in visited:
                    BFS_queue.append(node_bfs)
                    visited.append(node_bfs)
            if len(BFS_queue) == 0:
                visited = [start_node]
                BFS_queue = [start_node]

        #walk_ = walk_.pop(0)
        return walk_

    """
    Specific function of bfs sampling for max pool operation
    """
    def BFS_sample(self,start_node):
        walk_ = []
        visited = [start_node]
        #print(start_node)
        BFS_queue = [start_node]
        # neighborhood = get_neighborhood_data
        while len(walk_) < self.neighborhood_sample_num:
            #print("Im here")
            #print(BFS_queue.pop(0))
            cur = np.int(BFS_queue.pop(0))
            walk_.append(cur)
            cur_nbrs = sorted(self.get_neighborhood_data(cur)[0])
            for node_bfs in cur_nbrs:
                if not node_bfs in visited:
                    BFS_queue.append(node_bfs)
                    visited.append(node_bfs)
            if len(BFS_queue) == 0:
                visited = [start_node]
                BFS_queue = [start_node]

        #walk_ = walk_.pop(0)
        return walk_


    """
    compute average for one neighborhood node
    """


    def average_neighborhood(self, node_index, center_neighbor_size):
        neighbor_vec = self.assign_value(node_index)
        neighbor, neighbor_size = self.get_neighborhood_data(node_index)
        average_factor = 1 / np.sqrt(neighbor_size * center_neighbor_size)

        return neighbor_vec * average_factor


    """
    GCN Neighborhood extractor
    """


    def GCN_aggregator(self, node_index):
        neighbors, size = self.get_neighborhood_data(node_index)
        aggregate_vector = np.zeros(self.attribute_size)
        for index in neighbors:
            neighbor_average_vec = self.average_neighborhood(index, size)
            aggregate_vector += neighbor_average_vec

        return aggregate_vector


    """
    mean_pooling neighborhood
    """


    def mean_pooling_neighbor(self, node_index):
        neighbors, size = self.get_neighborhood_data(node_index)
        attribute_vector_total = np.zeros(self.attribute_size)
        for index in neighbors:
            attribute_vector_total += self.assign_value(index)

        return attribute_vector_total / size

    """
    max_pooling neighbor
    """
    def max_pooling_neighbor(self,node_index):
        samples = self.BFS_sample(node_index)
        max_pool_att = np.zeros((self.neighborhood_sample_num,self.attribute_size))
        k = 0
        for i in samples:
            one_att = self.assign_value(i)
            max_pool_att[k,:] = one_att
            k = k + 1

        return max_pool_att


    """
    Define get batch 
    """


    def get_batch_BFS(self,start_index):
        walk = np.zeros((self.batch_size, self.walk_length))
        batch_start_nodes = []
        nodes = np.array(self.G.nodes())
        for i in range(self.batch_size):
            #walk_single = np.array(self.BFS_search(nodes[i + start_index]))
            if self.option_lp_nc == 1:
                """
                task for link prediction
                """
                walk_single = np.array(self.walk_(nodes[i + start_index]))
                batch_start_nodes.append(nodes[i + start_index])
            if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                """
                task for node classification
                """
                walk_single = np.array(self.walk_(self.train_nodes[i + start_index]))
                batch_start_nodes.append(self.train_nodes[i + start_index])
            walk[i, :] = walk_single
        return walk, batch_start_nodes


    """
    get minibatch center data
    """


    def get_minibatch(self, index_vector):
        mini_batch = np.zeros((self.batch_size, self.attribute_size))
        index = 0
        for node_index in index_vector:
            x_center1 = self.assign_value(node_index)
            mini_batch[index, :] = x_center1
            index += 1

        return mini_batch


    """
    get batch neighbor_GCN_aggregate
    """


    def get_batch_GCNagg(self, index_vector):
        mini_batch_gcn_agg = np.zeros((self.batch_size, self.attribute_size))
        index = 0
        for node_index in index_vector:
            single_gcn = self.GCN_aggregator(node_index)
            mini_batch_gcn_agg[index, :] = single_gcn
            index += 1

        return mini_batch_gcn_agg

    """
    get batch neighbor mean
    """
    def get_batch_mean_pooling_neighbor(self,index_vector):
        mini_batch_mean_agg = np.zeros((self.batch_size,self.attribute_size))
        index = 0
        for node_index in index_vector:
            single_mean_pool = self.mean_pooling_neighbor(node_index)
            mini_batch_mean_agg[index, :] = single_mean_pool
            index += 1

        return mini_batch_mean_agg

    """
    get batch max pooling
    """
    def get_batch_max_pooling(self,index_vector):
        mini_batch_max_agg = np.zeros((self.batch_size, self.neighborhood_sample_num, self.attribute_size))
        index = 0
        for node_index in index_vector:
            single_max_pool = self.max_pooling_neighbor(node_index)
            mini_batch_max_agg[index, :, :] = single_max_pool
            index += 1

        return mini_batch_max_agg

    """
    get batch n2v
    """

    def get_batch_n2v(self, index_vector):
        mini_batch_n2v = np.zeros((self.batch_size,self.length))
        index = 0
        for node in index_vector:
            single_n2v = self.assign_value_n2v(node)
            mini_batch_n2v[index, :] = single_n2v
            index += 1

        return mini_batch_n2v

    #def get_batch_n2v(self,index_vector):
     #   mini_batch_n2v = np.zeros((self.batch_size,self.length))

    """
    get batch negative sampling for attritbute
    """


    def get_batch_negative(self,negative_samples):
        mini_batch_negative = np.zeros((self.batch_size, self.negative_sample_size, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in negative_samples[i, :]:
                negative_sample = self.GCN_aggregator(node)
                mini_batch_negative[i, index, :] = negative_sample
                index += 1
        return mini_batch_negative

    """
    get batch negative sampling for n2v
    """

    def get_batch_negative_n2v(self,negative_samples):
        mini_batch_negative = np.zeros((self.batch_size, self.negative_sample_size, self.length))
        for i in range(self.batch_size):
            index = 0
            for node in negative_samples[i, :]:
                negative_sample = self.assign_value_n2v(node)
                mini_batch_negative[i, index, :] = negative_sample
                index += 1
        return mini_batch_negative
    """
    get batch negative sampling for center node
    """

    def get_batch_negative_center(self,negative_samples):
        mini_batch_negative = np.zeros((self.batch_size, self.negative_sample_size, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in negative_samples[i, :]:
                negative_sample = self.assign_value(node)
                mini_batch_negative[i, index, :] = negative_sample
                index += 1
        return mini_batch_negative

    """
    get batch negative sampling for meanpooling
    """
    def get_batch_negative_meanpool(self,negative_samples):
        mini_batch_negative = np.zeros((self.batch_size, self.negative_sample_size, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in negative_samples[i, :]:
                negative_sample = self.mean_pooling_neighbor(node)
                mini_batch_negative[i, index, :] = negative_sample
                index += 1
        return mini_batch_negative

    """
    get batch negative sampling for maxpooling
    """
    def get_batch_negative_maxpooling(self,negative_samples):
        mini_batch_negative = np.zeros((self.batch_size, self.negative_sample_size, self.neighborhood_sample_num,
                                        self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in negative_samples[i, :]:
                negative_sample = self.max_pooling_neighbor(node)
                mini_batch_negative[i, index, :, :] = negative_sample
                index += 1
        return mini_batch_negative



    """
    get batch skip_gram samples for attribute
    """


    def get_batch_skip_gram(self, skip_gram_vecs):
        mini_batch_skip_gram = np.zeros((self.batch_size, self.walk_length, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in skip_gram_vecs[i, :]:
                skip_gram_sample = self.GCN_aggregator(node)
                mini_batch_skip_gram[i, index, :] = skip_gram_sample
                index += 1
        return mini_batch_skip_gram

    """
    get batch skip_gram samples for n2v
    """
    def get_batch_skip_gram_n2v(self, skip_gram_vecs):
        mini_batch_skip_gram = np.zeros((self.batch_size, self.walk_length, self.length))
        for i in range(self.batch_size):
            index = 0
            for node in skip_gram_vecs[i, :]:
                skip_gram_sample = self.assign_value_n2v(node)
                mini_batch_skip_gram[i, index, :] = skip_gram_sample
                index += 1
        return mini_batch_skip_gram

    """
    get batch for center nodes skip_gram
    """
    def get_batch_skip_gram_center(self,skip_gram_vecs):
        mini_batch_skip_gram = np.zeros((self.batch_size,self.walk_length,self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in skip_gram_vecs[i,:]:
                skip_gram_sample = self.assign_value(node)
                mini_batch_skip_gram[i,index,:] = skip_gram_sample
                index += 1
        return mini_batch_skip_gram
    """
    get batch for mean pooling skip_gram
    """
    def get_batch_skip_gram_meanpool(self,skip_gram_vecs):
        mini_batch_skip_gram = np.zeros((self.batch_size, self.walk_length, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in skip_gram_vecs[i, :]:
                skip_gram_sample = self.mean_pooling_neighbor(node)
                mini_batch_skip_gram[i, index, :] = skip_gram_sample
                index += 1
        return mini_batch_skip_gram

    """
    get batch for max pooling skip_gram
    """
    def get_batch_skip_gram_maxpool(self,skip_gram_vecs):
        mini_batch_skip_gram = np.zeros((self.batch_size, self.walk_length, self.neighborhood_sample_num,
                                         self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in skip_gram_vecs[i, :]:
                skip_gram_sample = self.max_pooling_neighbor(node)
                mini_batch_skip_gram[i, index, :, :] = skip_gram_sample
                index += 1
        return mini_batch_skip_gram


    """
    Uniform sample negative data
    """


    def uniform_get_negative_sample(self, skip_gram_vec, center_node):
        negative_samples = []
        #node_neighbors, neighbor_size = self.get_neighborhood_data(center_node)
        total_negative_samples = 0
        #print("im here in neg")
        negative_candidates = [x for x in self.dic_non_edges[center_node] if x not in skip_gram_vec]
        #print("finished neg")
        while (total_negative_samples < self.negative_sample_size):
            index_sample = np.int(np.floor(np.random.uniform(0, len(negative_candidates), 1)))
            sample = np.int(np.array(negative_candidates)[index_sample])
            negative_samples.append(sample)
            total_negative_samples += 1
        """
            index_sample = np.int(np.floor(np.random.uniform(0, np.array(self.G.nodes()).shape[0], 1)))
            sample = np.int(np.array(self.G.nodes())[index_sample])
            correct_negative_sample = 1
            for positive_sample in skip_gram_vec:
                if sample == np.int(positive_sample):
                    correct_negative_sample = 0
                    break
            if correct_negative_sample == 0:
                continue
            for neighborhood_sample in node_neighbors:
                if sample == neighborhood_sample:
                    correct_negative_sample = 0
            if sample == center_node:
                correct_negative_sample = 0
            if correct_negative_sample == 1:
                total_negative_samples += 1
                negative_samples.append(sample)
        """
        #print("finished while")

        return negative_samples


    """
    get batch mean_pooling
    """


    def get_batch_mean_pooling(self, index_vector):
        mini_batch_mean_pooling = np.zeros((self.batch_size, self.attribute_size))
        mini_batch_skip_gram_vectors = np.zeros((self.batch_size, self.walk_length))
        index = 0
        for node_index in index_vector:
            # skip_gram_vector = np.array(get_attribute_data_skip_gram(node_index, walk_length))
            skip_gram_vector = np.array(self.BFS_search(node_index))
            y_mean_pooling1 = self.mean_pooling(skip_gram_vector)
            mini_batch_mean_pooling[index, :] = y_mean_pooling1
            mini_batch_skip_gram_vectors[index, :] = skip_gram_vector
            index += 1

        return mini_batch_mean_pooling, mini_batch_skip_gram_vectors



    def get_data(self, start_index):
        # mini_batch_raw = np.array(Node2vec.simulate_walks(batch_size,walk_length))
        mini_batch_raw, start_nodes = get_batch_BFS(start_index)
        negative_samples = np.zeros((self.batch_size, self.negative_sample_size))
        negative_samples_vectors = np.zeros((self.batch_size, self.negative_sample_size, 8))
        skip_gram_vectors = np.zeros((self.batch_size, self.walk_length, 8))
        mini_batch_y = np.zeros((self.batch_size, self.length))
        mini_batch_x_label = np.zeros((self.batch_size, self.length))
        mini_batch_x_y = np.zeros((self.batch_size, self.length * 2))
        mini_batch_x = self.get_minibatch(start_nodes)
        batch_GCN_agg = self.get_batch_GCNagg(start_nodes)
        mini_batch_y_mean_pool, mini_batch_skip_gram = self.get_batch_mean_pooling(start_nodes)
        #batch_b_concat = convert_binary(mini_batch_y_mean_pool)

        # for i in range(batch_size):
        # negative_samples[i,:] = uniform_get_negative_sample(G,mini_batch_raw[i,:],start_nodes[i],negative_sample_size)

        # negative_samples_vectors = get_batch_negative(G,negative_samples,batch_size,negative_sample_size)

        # skip_gram_vectors = get_batch_skip_gram(G,mini_batch_raw,batch_size,walk_length)

        for i in range(self.batch_size):
            # index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
            # mini_batch_x[i,:] = 1
            # prob = 1/walk_length
            for j in range(self.walk_length):
                indexy = self.G.nodes[mini_batch_raw[i][j]]['index']
                mini_batch_y[i][indexy] = 1 / walk_length  # += prob
            # mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
            indexx = self.G.nodes[start_nodes[i]]['index']
            mini_batch_x_label[i][indexx] = 1
            mini_batch_x_y[i] = np.concatenate((mini_batch_x_label[i], mini_batch_y[i]), axis=None)

        mini_batch_concat_x_y = np.concatenate((mini_batch_x, mini_batch_y_mean_pool), axis=1)

        return mini_batch_x, mini_batch_y, batch_GCN_agg, negative_samples_vectors, skip_gram_vectors, mini_batch_x_label, mini_batch_x_y, mini_batch_y_mean_pool, mini_batch_concat_x_y


    def get_data_one_batch(self, start_index_):
        mini_batch_integral = np.zeros((self.batch_size, 1 + self.walk_length + self.negative_sample_size, self.attribute_size))
        mini_batch_integral_n2v = np.zeros((self.batch_size, 1 + self.walk_length + self.negative_sample_size, self.data_length))
        mini_batch_integral_centers = np.zeros((self.batch_size, 1+self.walk_length+self.negative_sample_size,self.attribute_size))
        mini_batch_integral_mean_agg = np.zeros((self.batch_size, 1+self.walk_length+self.negative_sample_size,self.attribute_size))
        mini_batch_integral_maxpool = np.zeros((self.batch_size, 1+self.walk_length+self.negative_sample_size,self.neighborhood_sample_num,
                                                self.attribute_size))
        mini_batch_raw, start_nodes = self.get_batch_BFS(start_index_)
        #mini_batch_y = np.zeros((self.batch_size, self.data_length))
        #mini_batch_x_label = np.zeros((self.batch_size, self.data_length))
        mini_batch_y_label = np.zeros((self.batch_size, self.class_num))
        #mini_batch_y_mean_pool = np.zeros(3)
        """
        get different batch set for different model
        """
        batch_center_x = self.get_minibatch(start_nodes)
        if self.option == 5:
            batch_mean_agg = self.get_batch_mean_pooling_neighbor(start_nodes)
        if self.option == 1 or self.option == 3:
            batch_GCN_agg = self.get_batch_GCNagg(start_nodes)
        if not self.option == 1:
            batch_n2v = self.get_batch_n2v(start_nodes)
        if self.option == 6:
            batch_maxpool_agg = self.get_batch_max_pooling(start_nodes)
        negative_samples = np.zeros((self.batch_size, self.negative_sample_size))
        #skip_gram_vectors = np.zeros((batch_size, walk_length, attribute_size))
        #negative_samples_vectors = np.zeros((batch_size, negative_sample_size, attribute_size))
        for i in range(self.batch_size):
            if self.option == 1 or self.option == 3:
                mini_batch_integral[i, 0, :] = batch_GCN_agg[i, :]
            #indexy = self.G.node[start_nodes[i]]['node_index']
            if not self.option == 1:
                mini_batch_integral_n2v[i, 0, :] = batch_n2v[i,:]
            mini_batch_integral_centers[i,0,:] = batch_center_x[i,:]
            if self.option == 5:
                mini_batch_integral_mean_agg[i,0,:] = batch_mean_agg[i,:]
            if self.option == 6:
                mini_batch_integral_maxpool[i,0,:,:] = batch_maxpool_agg[i,:,:]

        #mini_batch_y_mean_pool, mini_batch_skip_gram = self.get_batch_mean_pooling(start_nodes)
    
        for i in range(self.batch_size):
            negative_samples[i, :] = self.uniform_get_negative_sample(mini_batch_raw[i, :], start_nodes[i])
        if self.option == 1 or self.option == 3:
            negative_samples_vectors = self.get_batch_negative(negative_samples)
        if not self.option == 1:
            negative_samples_vectors_n2v = self.get_batch_negative_n2v(negative_samples)
        negative_samples_vectors_center = self.get_batch_negative_center(negative_samples)
        if self.option == 5:
            negative_samples_vectors_meanpool = self.get_batch_negative_meanpool(negative_samples)
        if self.option == 6:
            negative_samples_vectors_maxpool = self.get_batch_negative_maxpooling(negative_samples)

        if self.option == 1 or self.option == 3:
            skip_gram_vectors = self.get_batch_skip_gram(mini_batch_raw)
        if not self.option == 1:
            skip_gram_vectors_n2v = self.get_batch_skip_gram_n2v(mini_batch_raw)
        skip_gram_vectors_center =self.get_batch_skip_gram_center(mini_batch_raw)
        if self.option == 5:
            skip_gram_vectors_meanpool = self.get_batch_skip_gram_meanpool(mini_batch_raw)
        if self.option == 6:
            skip_gram_vectors_maxpool = self.get_batch_skip_gram_maxpool(mini_batch_raw)
        for i in range(self.batch_size):
            if self.option == 1 or self.option == 3:
                mini_batch_integral[i, 1:self.walk_length + 1, :] = skip_gram_vectors[i, :, :]
            if not self.option == 1:
                mini_batch_integral_n2v[i, 1:self.walk_length + 1, :] = skip_gram_vectors_n2v[i, :, :]
            mini_batch_integral_centers[i,1:self.walk_length+1,:] = skip_gram_vectors_center[i,:,:]
            if self.option == 5:
                mini_batch_integral_mean_agg[i, 1:self.walk_length + 1, :] = skip_gram_vectors_meanpool[i, :, :]
            if self.option == 6:
                mini_batch_integral_maxpool[i, 1:self.walk_length + 1, :,:] = skip_gram_vectors_maxpool[i,:,:,:]
    
        for i in range(self.batch_size):
            if self.option == 1 or self.option == 3:
                mini_batch_integral[i, self.walk_length + 1:, :] = negative_samples_vectors[i, :, :]
            if not self.option == 1:
                mini_batch_integral_n2v[i,self.walk_length + 1:, :] = negative_samples_vectors_n2v[i, :, :]
            mini_batch_integral_centers[i,self.walk_length+1:,:] = negative_samples_vectors_center[i,:,:]
            if self.option == 5:
                mini_batch_integral_mean_agg[i, self.walk_length + 1:, :] = negative_samples_vectors_meanpool[i, :, :]
            if self.option == 6:
                mini_batch_integral_maxpool[i, self.walk_length + 1:, :,:] = negative_samples_vectors_maxpool[i,:,:]
    
        """
        for i in range(batch_size):
          #index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
          #mini_batch_x[i,:] = 1
          #prob = 1/walk_length
          for j in range(walk_length):
            indexy = G.nodes[mini_batch_raw[i][j]]['node_index']
            mini_batch_y[i][indexy] = 1#/walk_length#+= prob
          #mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
          indexx = G.nodes[start_nodes[i]]['node_index']
          mini_batch_x_label[i][indexx] = 1
        """
        """
        for i in range(self.batch_size):
            indexy = self.G.node[start_nodes[i]]['node_index']
            mini_batch_y[i][indexy] = 1
            for j in self.G.neighbors(start_nodes[i]):
                indexy = self.G.node[j]['node_index']
                mini_batch_y[i][indexy] = 1
        """

        for i in range(self.batch_size):
            indexy = self.G.node[start_nodes[i]]['label']
            mini_batch_y_label[i][indexy] = 1

        return mini_batch_integral, mini_batch_integral_n2v, mini_batch_integral_centers, \
               mini_batch_integral_mean_agg, mini_batch_integral_maxpool, mini_batch_y_label

    def train(self):
        if self.option_lp_nc == 1:
            G_num = len(self.G.nodes())

        if self.option_lp_nc == 2 or self.option_lp_nc == 3:
            G_num = len(self.train_nodes)
        iter_num = np.int(np.floor(G_num/self.batch_size))
        k = 0
        epoch = 2

        while(k<epoch):

            print("training in epoch")
            print(k)
            for j in range(iter_num):
                start_time = time.time()
                sample_index = j*self.batch_size #np.int(np.floor(np.random.uniform(0, )))
                mini_batch_integral, mini_batch_integral_n2v, mini_batch_integral_centers, \
                mini_batch_integral_mean_agg,mini_batch_integral_maxpool, mini_batch_y_label = \
                    self.get_data_one_batch(sample_index)
                if self.option == 1:
                    print("running mf_gcn")
                    err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict=
                                                                             {self.x_gcn: mini_batch_integral,
                                                                              self.y_label:mini_batch_y_label})
                    print(err_[0])
                    if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                        err_sup = self.sess.run([self.cross_entropy, self.train_step_cross_entropy], feed_dict=
                                                                                {self.x_gcn: mini_batch_integral,
                                                                                 self.y_label: mini_batch_y_label
                                                                                 })
                        print(err_sup[0])
                if self.option == 2:
                    print("running n2v")
                    err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict=
                                                                            {self.x_n2v: mini_batch_integral_n2v,
                                                                             self.y_label: mini_batch_y_label})
                    print(err_[0])
                    if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                        err_sup = self.sess.run([self.cross_entropy, self.train_step_cross_entropy], feed_dict=
                                                                                {self.x_n2v: mini_batch_integral_n2v,
                                                                                 self.y_label: mini_batch_y_label})
                        print(err_sup[0])
                if self.option == 3:
                    print("running structure+feature")
                    err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict=
                                                                            {self.x_n2v: mini_batch_integral_n2v,
                                                                             self.x_gcn: mini_batch_integral,
                                                                             self.x_center: mini_batch_integral_centers,
                                                                             self.y_label:mini_batch_y_label})
                    print(err_[0])
                    if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                        err_sup = self.sess.run([self.cross_entropy, self.train_step_cross_entropy], feed_dict=
                                                                                {self.x_n2v: mini_batch_integral_n2v,
                                                                                 self.x_gcn: mini_batch_integral,
                                                                                 self.x_center: mini_batch_integral_centers,
                                                                                 self.y_label: mini_batch_y_label
                                                                                 })
                        print(err_sup[0])

                if self.option == 4:
                    print("running structure+raw_feature")
                    err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict=
                                                                            {self.x_n2v: mini_batch_integral_n2v,
                                                                             self.x_center: mini_batch_integral_centers,
                                                                             self.y_label: mini_batch_y_label})
                    print(err_[0])
                    if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                        err_sup = self.sess.run([self.cross_entropy, self.train_step_cross_entropy], feed_dict=
                                                                                {self.x_n2v: mini_batch_integral_n2v,
                                                                                 self.x_center: mini_batch_integral_centers,
                                                                                 self.y_label: mini_batch_y_label
                                                                                 })
                        print(err_sup[0])

                if self.option == 5:
                    print("running structure+graphsage_mean_pool")
                    err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict=
                                                                            {self.x_n2v: mini_batch_integral_n2v,
                                                                             self.x_center: mini_batch_integral_centers,
                                                                             self.x_mean_pool:mini_batch_integral_mean_agg,
                                                                             self.y_label: mini_batch_y_label})
                    print(err_[0])
                    if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                        err_sup = self.sess.run([self.cross_entropy, self.train_step_cross_entropy], feed_dict=
                                                                                {self.x_n2v: mini_batch_integral_n2v,
                                                                                 self.x_center: mini_batch_integral_centers,
                                                                                 self.x_mean_pool:mini_batch_integral_mean_agg,
                                                                                 self.y_label: mini_batch_y_label
                                                                                 })
                        print(err_sup[0])

                if self.option == 6:
                    print("running graphsage maxpool")
                    err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict=
                                                                            {self.x_n2v: mini_batch_integral_n2v,
                                                                             self.x_center: mini_batch_integral_centers,
                                                                             self.x_max_pool: mini_batch_integral_maxpool,
                                                                             self.y_label: mini_batch_y_label})
                    print(err_[0])
                    if self.option_lp_nc == 2 or self.option_lp_nc == 3:
                        err_sup = self.sess.run([self.cross_entropy, self.train_step_cross_entropy], feed_dict=
                                                                                {self.x_n2v: mini_batch_integral_n2v,
                                                                                 self.x_center: mini_batch_integral_centers,
                                                                                 self.x_max_pool:mini_batch_integral_maxpool,
                                                                                 self.y_label: mini_batch_y_label
                                                                                 })
                        print(err_sup[0])

                print("one iteration uses %s seconds" % (time.time() - start_time))
            k = k + 1
