import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class hetero_model():
    """
    Create deep heterogenous embedding model, by using metapath learning and TransE model.
    """
    def __init__(self,kg):
        self.batch_size = 64
        self.walk_length_iter = 5
        self.latent_dim = 100
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size =len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.kg = kg

        self.shape_relation = (self.latent_dim, )
        self.init_test = tf.keras.initializers.he_normal(seed=None)
        self.init_diag = tf.keras.initializers.he_normal(seed=None)
        """
        initial relation type test
        """
        self.relation_test = tf.Variable(self.init_test(shape=self.shape_relation))

        """
        initial relation type diag
        """
        self.relation_diag = tf.Variable(self.init_diag(shape=self.shape_relation))

        """
        define test vector
        """
        self.item = tf.placeholder(tf.float32, [None, self.item_size])

        """
        define patient vector
        """
        self.patient = tf.placeholder(tf.float32,[None, self.patient_size])

        """
        define diagnosis vector
        """
        self.diagnosis = tf.placeholder(tf.float32,[None, self.diagnosis_size])

        """
        Create meta-path type
        """
        self.meta_path1 = ['I','P','D','P','I']
        self.meta_path2 = ['P','D','P']
        self.meta_path3 = ['I','P','I']
        self.meta_path4 = ['P','I','P']
        self.meta_path5 = ['D','P','D']
        self.meta_path6 = ['D','P','I','P','D']

    def generate_next_node(self, node_type, index, meta_path):
        """
        generate next move based on current node type
        and current node index
        """
        meta_path = meta_path
        walk = []
        walk.append([node_type,index])
        cur_index = index
        cur_node_type = node_type
        meta_path.pop(0)
        for i in meta_path:
            if i == 'P':
                if cur_node_type == 'item':
                    neighbor = self.kg.dic_item[cur_index]['neighbor_patient']
                    """
                    uniformly generate sampling number
                    """
                    random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                    cur_index = neighbor[random_index]
                    cur_node_type = 'patient'
                    walk.append([cur_node_type,cur_index])
                if cur_node_type == 'diagnosis':
                    neighbor = self.kg.dic_diag[cur_index]['neighbor_patient']
                    """
                    uniformly generate sampling number
                    """
                    random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                    cur_index = neighbor['neighbor_patient'][random_index]
                    cur_node_type = 'patient'
                    walk.append([cur_node_type,cur_index])
            if i == "D":
                if cur_node_type == 'patient':
                    neighbor = self.kg.dic_patient[cur_index]['neighbor_diag']
                    """
                    uniformly generate sampling number
                    """
                    random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                    cur_index = neighbor[random_index]
                    cur_node_type = 'diagnosis'
                    walk.append([cur_node_type, cur_index])
            if i == "I":
                if cur_node_type == 'patient':
                    neighbor = list(self.kg.dic_patient[cur_index]['itemid'].keys())
                    """
                    uniformly generate sampling number
                    """
                    random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                    cur_index = neighbor[random_index]
                    cur_node_type = 'item'
                    walk.append([cur_node_type, cur_index])

        return walk



    def extract_meta_path(self, node_type, start_index):
        """
        Perform metapath from different starting node type
        node_type: node_type
        start_index: ID for starting node
        meta path
        """
        walk = []
        for i in range(self.walk_length_iter):
            walk += self.generate_next_node(node_type, start_index, self.meta_path1)

        return walk



