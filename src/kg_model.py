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
        self.negative_sample_size = 30
        self.prop_neg = 0.1
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
        self.patient = tf.placeholder(tf.float32,[None, self.item_size])

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

    def build_hetero_model(self):
        """
        build heterogenous graph learning model
        """

        """
        build item projection layer
        """
        self.Dense_item = tf.layers.dense(inputs=self.item,
                                           units=self.latent_dim,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        """
        build diagnosis projection layer
        """
        self.Dense_diag = tf.layers.dense(inputs=self.diagnosis,
                                           units=self.latent_dim,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)

        """
        build patient projection layer
        """
        self.Dense_patient = tf.layers.dense(inputs=self.patient,
                                           units=self.latent_dim,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)


    def generate_next_node(self, node_type, index, meta_path):
        """
        generate next move based on current node type
        and current node index
        """
        meta_path_ = meta_path[1:]
        walk = []
        #walk.append([node_type,index])
        cur_index = index
        cur_node_type = node_type
        for i in meta_path_:
            if i == 'P':
                if cur_node_type == 'item':
                    neighbor = list(self.kg.dic_item[cur_index]['neighbor_patient'])
                    """
                    uniformly generate sampling number
                    """
                    random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                    cur_index = neighbor[random_index]
                    cur_node_type = 'patient'
                    walk.append([cur_node_type,cur_index])
                if cur_node_type == 'diagnosis':
                    neighbor = list(self.kg.dic_diag[cur_index]['neighbor_patient'])
                    """
                    uniformly generate sampling number
                    """
                    random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                    cur_index = neighbor[random_index]
                    cur_node_type = 'patient'
                    walk.append([cur_node_type,cur_index])
            if i == "D":
                if cur_node_type == 'patient':
                    neighbor = list(self.kg.dic_patient[cur_index]['neighbor_diag'])
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

    """
    assign value to one patient sample
    """
    def assign_value_patient(self, patientid):
        one_sample = np.zeros(self.item_size)
        for i in self.kg.dic_patient[patientid]['itemid'].keys():
            ave_value = np.mean(self.kg.dic_patient[patientid]['itemid'][i])
            index = self.kg.dic_item[i]['item_index']
            one_sample[index] = ave_value

        return one_sample

    """
    assign value to one item sample
    """
    def assign_value_item(self, itemid):
        one_sample = np.zeros(self.item_size)
        index = self.kg.dic_item[itemid]['item_index']
        one_sample[index] = 1

        return one_sample

    """
    assign value to one diagnosis sample
    """
    def assign_value_diag(self,diagid):
        one_sample = np.zeros(self.diagnosis_size)
        index = self.kg.dic_diag[diagid]['diag_index']
        one_sample[index] = 1

        return one_sample

    """
    prepare data for one metapath
    """
    def get_positive_sample_metapath(self, meta_path):
        patient_nodes = []
        item_nodes = []
        diag_nodes = []
        for i in meta_path:
            if i[0] == 'patient':
                patient_id = i[1]
                patient_sample = self.assign_value_patient(patient_id)
                patient_nodes.append(patient_sample)
            if i[0] == 'item':
                item_id = i[1]
                item_sample = self.assign_value_item(item_id)
                item_nodes.append(item_sample)
            if i[0] == 'diagnosis':
                diag_id = i[1]
                diag_sample = self.assign_value_diag(diag_id)
                diag_nodes.append(diag_sample)

        return np.array(patient_nodes), np.array(item_nodes), np.array(diag_nodes)

    """
    prepare data for one metapath negative sample 
    """
    def get_negative_sample_metapath(self,center_node_type,center_node_index):
        patient_neg_sample = None
        item_neg_sample = None
        diag_neg_sample = None
        neg_nodes_patient, neg_nodes_item, neg_nodes_diag = \
            self.get_negative_samples(center_node_type,center_node_index)

        length_patient = len(neg_nodes_patient)
        length_item = len(neg_nodes_item)
        length_diag = len(neg_nodes_diag)

        if not neg_nodes_patient == []:
            index = 0
            patient_neg_sample = np.zeros((length_patient, self.item_size))
            for i in neg_nodes_patient:
                one_sample_neg_patient = self.assign_value_patient(i)
                patient_neg_sample[index,:] = one_sample_neg_patient
                index += 1

        if not neg_nodes_item == []:
            index = 0
            item_neg_sample = np.zeros((length_item, self.item_size))
            for i in neg_nodes_item:
                one_sample_neg_item = self.assign_value_item(i)
                item_neg_sample[index,:] = one_sample_neg_item
                index += 1

        if not neg_nodes_diag == []:
            index = 0
            diag_neg_sample = np.zeros((length_diag, self.diagnosis_size))
            for i in neg_nodes_diag:
                one_sample_neg_diag = self.assign_value_diag(i)
                diag_neg_sample[index,:] = one_sample_neg_diag
                index += 1

        return patient_neg_sample, item_neg_sample, diag_neg_sample

    """
    prepare data for negative heterougenous sampling
    """
    def get_negative_samples(self,center_node_type,center_node_index):
        neg_nodes_patient = []
        neg_nodes_item = []
        neg_nodes_diag = []
        if center_node_type == 'patient':
            """
            get neg set for item
            """
            item_neighbor_nodes = self.kg.dic_patient[center_node_index]['itemid'].keys()
            whole_item_nodes = self.kg.dic_item.keys()
            neg_set_item = [i for i in whole_item_nodes if i not in item_neighbor_nodes]
            for j in range(self.negative_sample_size / 2):
                index_sample = np.int(np.floor(np.random.uniform(0, len(neg_set_item), 1)))
                neg_nodes_item.append(neg_set_item[index_sample])

            """
            get neg set for diag
            """
            diag_neighbor_nodes = self.kg.dic_patient[center_node_index]['neighbor_diag']
            whole_diag_nodes = self.kg.dic_diag.keys()
            neg_set_diag = [i for i in whole_diag_nodes if i not in diag_neighbor_nodes]
            for j in range(self.negative_sample_size / 2):
                index_sample = np.int(np.floor(np.random.uniform(0, len(neg_set_diag), 1)))
                neg_nodes_diag.append(neg_set_diag[index_sample])

        if center_node_type == 'item':
            """
            get neg set for patient
            """
            patient_neighbor_nodes = self.kg.dic_item[center_node_index]['neighbor_patient']
            whole_patient_nodes = self.kg.dic_patient.keys()
            neg_set_patient = [i for i in whole_patient_nodes if i not in patient_neighbor_nodes]
            for j in range(self.negative_sample_size):
                index_sample = np.int(np.floor(np.random.uniform(0, len(neg_set_patient), 1)))
                neg_nodes_patient.append(neg_set_patient[index_sample])

        if center_node_type == 'diagnosis':
            """
            get neg set for patient
            """
            patient_neighbor_nodes = self.kg.dic_diag[center_node_index]['neighbor_patient']
            whole_patient_nodes = self.kg.dic_patient.keys()
            neg_set_patient = [i for i in whole_patient_nodes if i not in patient_neighbor_nodes]
            for j in range(self.negative_sample_size):
                index_sample = np.int(np.floor(np.random.uniform(0, len(neg_set_patient), 1)))
                neg_nodes_patient.append(neg_set_patient[index_sample])

        return neg_nodes_patient, neg_nodes_item, neg_nodes_diag





    def extract_meta_path(self, node_type, start_index, meta_path_type):
        """
        Perform metapath from different starting node type
        node_type: node_type
        start_index: ID for starting node
        meta path
        """
        walk = []
        walk.append([node_type,start_index])
        for i in range(self.walk_length_iter):
            cur_node_type = walk[-1][0]
            cur_index = walk[-1][1]
            walk += self.generate_next_node(cur_node_type, cur_index, meta_path_type)

        return walk





