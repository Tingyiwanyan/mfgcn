import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby

class hetero_model():
    """
    Create deep heterogenous embedding model, by using metapath learning and TransE model.
    """
    def __init__(self,kg):
        self.batch_size = 16
        self.epoch = 6
        self.walk_length_iter = 4
        self.latent_dim = 100
        self.negative_sample_size = 30
        self.positive_sample_size = 24
        self.length_patient_pos = 12
        self.length_item_pos = 5
        self.length_diag_pos = 8
        self.length_patient_neg = 0
        self.length_item_neg = 15
        self.length_diag_neg = 15
        self.prop_neg = 0.1
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size =len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.kg = kg
        self.x_origin = None
        self.x_negative = None


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
        self.meta_path1 = ['I','P','D','P','D','P','I']
        self.meta_path2 = ['P','D','P']
        self.meta_path3 = ['I','P','I']
        self.meta_path4 = ['P','I','P']
        self.meta_path5 = ['D','P','D']
        self.meta_path6 = ['D','P','I','P','D']


    """
    configure model
    """
    def configure(self):
        self.build_hetero_model()
        self.get_latent_rep('patient')
        self.SGNN_loss()
        self.train_step_neg = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    """
    train model
    """
    def train(self):
        for i in range(self.epoch):
            print("epoch number")
            print(i)
            for j in self.kg.dic_patient.keys():
                single_patient, single_item, single_diag = self.get_one_data(self.meta_path1, 'patient', j)
                err_ = self.sess.run([self.negative_sum,self.train_step_neg], feed_dict= {self.patient: single_patient,
                                                                      self.item: single_item,
                                                                      self.diagnosis: single_diag})
                print(err_[0])

    """
    test model accuracy
    """
    def test(self, patient_id):
        self.patient_id_test = patient_id
        patient = np.zeros((1,self.item_size))
        patient[0,:] = self.assign_value_patient(patient_id)
        embed_patient = self.sess.run([self.Dense_patient],feed_dict={self.patient:patient})
        embed_patient_norm = embed_patient[0]/np.linalg.norm(embed_patient[0])
        self.embed_item = np.zeros((len(self.kg.dic_item),self.latent_dim))
        self.embed_diag = np.zeros((len(self.kg.dic_diag),self.latent_dim))
        self.pos_score_item = np.zeros(len(self.kg.dic_item.keys()))
        self.pos_score_diag = np.zeros(len(self.kg.dic_diag.keys()))
        for i in self.kg.dic_item.keys():
            index = self.kg.dic_item[i]['item_index']
            single_item = np.zeros((1,self.item_size))
            single_item[0,:] = self.assign_value_item(i)
            embed_single_item = self.sess.run([self.Dense_item],feed_dict={self.item:single_item})
            embed_item_single= embed_single_item[0]/np.linalg.norm(embed_single_item[0])
            self.embed_item[index,:] = embed_item_single
            self.pos_score_item[index] = np.sum(np.multiply(embed_patient_norm, embed_item_single))

        for i in self.kg.dic_diag.keys():
            index = self.kg.dic_diag[i]['diag_index']
            single_diag = np.zeros((1,self.diagnosis_size))
            single_diag[0,:] = self.assign_value_diag(i)
            embed_single_diag = self.sess.run([self.Dense_diag],feed_dict={self.diagnosis:single_diag})
            embed_diag_single = embed_single_diag[0]/np.linalg.norm(embed_single_diag[0])
            self.embed_diag[index,:] = embed_diag_single
            self.pos_score_diag[index] = np.sum(np.multiply(embed_patient_norm, embed_diag_single))

        self.seq_diag = sorted(self.pos_score_diag)
        self.seq_item = sorted(self.pos_score_item)
        self.seq_diag.reverse()
        self.seq_item.reverse()

        self.index_diag = [list(self.pos_score_diag).index(v) for v in self.seq_diag]
        self.index_item = [list(self.pos_score_item).index(v) for v in self.seq_item]

        correct_detect_num_item = 0
        num_item = len(self.kg.dic_patient[patient_id]['itemid'].keys())
        detect_index_item = np.array(np.where(self.pos_score_item>0.5))
        num_total_detect_item = len(detect_index_item)
        for j in self.kg.dic_patient[patient_id]['itemid'].keys():
            index_item = self.kg.dic_item[j]['item_index']
            if index_item in detect_index_item:
                correct_detect_num_item += 1

        pos_rate_item = correct_detect_num_item/np.float(num_item)

        correct_detect_num_diag = 0
        num_diag = len(self.kg.dic_patient[patient_id]['neighbor_diag'])
        detect_index_diag = np.array(np.where(self.pos_score_diag > 0.9))
        num_total_detect_diag = len(detect_index_diag)
        for j in self.kg.dic_patient[patient_id]['neighbor_diag']:
            index_diag = self.kg.dic_diag[j]['diag_index']
            if index_diag in detect_index_diag:
                correct_detect_num_diag += 1

        pos_rate_diag = correct_detect_num_diag / np.float(num_diag)


        return pos_rate_item, pos_rate_diag

    """
    get the ranked item, diag for one patient
    """
    def recommandation(self):
        length_diag = len(np.where(np.array(self.seq_diag)>0.9)[0])
        index_pick_diag = self.index_diag[0:length_diag]
        self.ICD = []
        for i in self.kg.dic_diag.keys():
            if self.kg.dic_diag[i]['diag_index'] in index_pick_diag:
                self.ICD.append(i)
        length_item = len(np.where(np.array(self.seq_item)>0.5)[0])
        self.item_test = []
        index_pick_item = self.index_item[0:length_item]
        for i in self.kg.dic_item.keys():
            if self.kg.dic_item[i]['item_index'] in index_pick_item:
                self.item_test.append(i)

        self.diagnosis_recom = []
        self.item_test_recom = []
        for i in range(len(self.ICD)):
            ICD_diag = self.ICD[i]
            if ICD_diag in self.kg.diag_d_ar[:,1]:
                index_ICD_diag = np.where(self.kg.diag_d_ar[:,1] == ICD_diag)[0][0]
                diagnosis = self.kg.diag_d_ar[index_ICD_diag][3]
                self.diagnosis_recom.append([ICD_diag,diagnosis])
        for i in range(len(self.item_test)):
            item_id = self.item_test[i]
            index_item_id = np.where(self.kg.d_item_ar[:,1] == item_id)[0][0]
            item_test_name = self.kg.d_item_ar[index_item_id][2]
            if item_id in self.kg.dic_patient[self.patient_id_test]['itemid'].keys():
                item_test_value = np.mean(self.kg.dic_patient[self.patient_id_test]['itemid'][item_id])
                self.item_test_recom.append([item_id,item_test_name,item_test_value])








    def build_hetero_model(self):
        """
        build heterogenous graph learning model
        """

        """
        build item projection layer
        """
        self.Dense_item_ = tf.layers.dense(inputs=self.item,
                                           units=self.latent_dim,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        """
        build diagnosis projection layer
        """
        self.Dense_diag_ = tf.layers.dense(inputs=self.diagnosis,
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

        self.Dense_item = tf.math.add(self.Dense_item_, self.relation_test)
        self.Dense_diag = tf.math.subtract(self.Dense_diag_, self.relation_diag)


    """
    get center node and positive & negative latent representation for one center node sample
    """
    def get_latent_rep(self,center_node_type):
        #meta_path_sample = self.extract_meta_path(center_node_type, center_node_index, meta_path_type)
        #patient_nodes, item_nodes, diag_nodes = get_positive_sample_metapath(meta_path_sample)
        self.x_skip_patient = None
        self.x_negative_patient = None
        self.x_skip_item = None
        self.x_negative_item = None
        self.x_skip_diag = None
        self.x_negative_diag = None
        if center_node_type == 'patient':
            idx_origin = tf.constant([0])
            self.x_origin = tf.gather(self.Dense_patient, idx_origin, axis=0)

        if center_node_type == 'item':
            idx_origin = tf.constant([0])
            self.x_origin = tf.gather(self.Dense_item, idx_origin, axis=0)

        if center_node_type == 'diagnosis':
            idx_origin = tf.constant([0])
            self.x_origin = tf.gather(self.Dense_diag, idx_origin, axis=0)

        """
        Case where center node is patient
        """
        if center_node_type == 'patient':
            if self.length_patient_pos > 1:
                patient_idx_skip = tf.constant([i + 1 for i in range(self.length_patient_pos-1)])
                self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=0)
            if not self.length_patient_neg == 0:
                patient_idx_negative = \
                    tf.constant([i + self.length_patient_pos for i in range(self.length_patient_neg)])
                self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=0)
            """
            getting positive samples
            """
            if not self.length_item_pos == 0:
                item_idx_skip = tf.constant([i for i in range(self.length_item_pos)])
                self.x_skip_item = tf.gather(self.Dense_item, item_idx_skip, axis=0)
            if not self.length_diag_pos == 0:
                diag_idx_skip = tf.constant([i for i in range(self.length_diag_pos)])
                self.x_skip_diag = tf.gather(self.Dense_diag, diag_idx_skip, axis=0)
            """
            getting negative samples
            """
            if not self.length_item_neg == 0:
                item_idx_negative = \
                    tf.constant([i + self.length_item_pos for i in range(self.length_item_neg)])
                self.x_negative_item = tf.gather(self.Dense_item, item_idx_negative, axis=0)
            if not self.length_diag_neg == 0:
                diag_idx_negative = \
                    tf.constant([i + self.length_diag_pos for i in range(self.length_diag_neg)])
                self.x_negative_diag = tf.gather(self.Dense_diag, diag_idx_negative, axis=0)

        """
        Case where center node is item
        """
        if center_node_type == 'item':
            if self.length_item_pos > 1:
                item_idx_skip = tf.constant([i + 1 for i in range(self.length_item_pos-1)])
                self.x_skip_item = tf.gather(self.Dense_item, item_idx_skip, axis=0)
            if not self.length_item_neg == 0:
                item_idx_negative = \
                    tf.constant([i + self.length_item_pos for i in range(self.length_item_neg)])
                self.x_negative_item = tf.gather(self.Dense_item, item_idx_negative, axis=0)
            """
            getting positive samples
            """
            if not self.length_patient_pos == 0:
                patient_idx_skip = tf.constant([i for i in range(self.length_patient_pos)])
                self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=0)
            if not self.length_diag_pos == 0:
                diag_idx_skip = tf.constant([i for i in range(self.length_diag_pos)])
                self.x_skip_diag = tf.gather(self.Dense_diag, diag_idx_skip, axis=0)
            """
            getting negative samples
            """
            if not self.length_patient_neg == 0:
                patient_idx_negative = \
                    tf.constant([i + self.length_patient_pos for i in range(self.length_patient_neg)])
                self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=0)
            if not self.length_diag_neg == 0:
                diag_idx_negative = \
                    tf.constant([i + self.length_diag_pos for i in range(self.length_diag_neg)])
                self.x_negative_diag = tf.gather(self.Dense_diag, diag_idx_negative, axis=0)

        """
        Case where center node is diag
        """
        if center_node_type == 'diagnosis':
            if self.length_diag_pos > 1:
                diag_idx_skip = tf.constant([i + 1 for i in range(self.length_diag_pos-1)])
                self.x_skip_diag = tf.gather(self.Dense_diag, diag_idx_skip, axis=0)
            if not self.length_diag_neg == 0:
                diag_idx_negative = \
                    tf.constant([i + self.length_diag_pos for i in range(self.length_diag_neg)])
                self.x_negative_diag = tf.gather(self.Dense_diag, diag_idx_negative, axis=0)
            """
            getting positive samples
            """
            if not self.length_patient_pos == 0:
                patient_idx_skip = tf.constant([i for i in range(self.length_item_pos)])
                self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=0)
            if not self.length_item_pos == 0:
                item_idx_skip = tf.constant([i for i in range(self.length_item_pos)])
                self.x_skip_item = tf.gather(self.Dense_item, item_idx_skip, axis=0)
            """
            getting negative samples
            """
            if not self.length_patient_neg == 0:
                patient_idx_negative = \
                    tf.constant([i + self.length_patient_pos for i in range(self.length_patient_neg)])
                self.x_negative_patient = tf.gather(self.Dense_patient, patient_idx_negative, axis=0)
            if not self.length_item_neg == 0:
                item_idx_negative = \
                    tf.constant([i + self.length_diag_pos for i in range(self.length_item_neg)])
                self.x_negative_item = tf.gather(self.Dense_item, item_idx_negative, axis=0)

        """
        combine skip samples and negative samples
        """
        self.x_skip = self.x_skip_patient
        self.x_skip = tf.concat([self.x_skip,self.x_skip_item],axis=0)
        self.x_skip = tf.concat([self.x_skip,self.x_skip_diag],axis=0)


        """
        prepare negative data
        """
        self.x_negative = self.x_negative_item
        self.x_negative = tf.concat([self.x_negative, self.x_negative_diag],axis=0)





    def SGNN_loss(self):
        """
        Implement sgnn with new structure
        """

        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=1)

        skip_training = tf.broadcast_to(self.x_origin, [self.negative_sample_size, self.latent_dim])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=1)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 1)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum))))

        positive_training = tf.broadcast_to(self.x_origin, [self.positive_sample_size, self.latent_dim])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=1)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=1)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 1)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive)))

        self.negative_sum = tf.math.negative(tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))


    def generate_next_node(self, node_type, index, meta_path):
        """
        generate next move based on current node type
        and current node index
        """
        walk = []
        walk.append([node_type,index])
        cur_index = index
        cur_node_type = node_type
        for i in meta_path:
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
            mean = self.kg.dic_item[i]['mean_value']
            std = self.kg.dic_item[i]['std']
            ave_value = np.mean(self.kg.dic_patient[patientid]['itemid'][i])
            index = self.kg.dic_item[i]['item_index']
            if std == 0:
                one_sample[index] = 0
            else:
                one_sample[index] = (np.float(ave_value)-mean)/std

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
    assign multi-hot diagnosis to one patient
    """
    def assign_multi_hot(self,patientid):
        one_sample = np.zeros(self.diagnosis_size)
        for i in self.kg.dic_patient[patientid]['neighbor_diag']:
            index = self.kg.dic_diag[i]['diag_index']
            one_sample[index] = 1

        return one_sample

    """
    preparing one data sample
    """
    def get_one_data(self,meta_path_type, center_node_type, center_node_index):
        single_meta_path = self.extract_meta_path(center_node_type, center_node_index, meta_path_type)
        patient_sample_pos, item_sample_pos, diag_sample_pos = \
            self.get_positive_sample_metapath(single_meta_path)
        patient_neg_sample, item_neg_sample, diag_neg_sample = \
            self.get_negative_sample_metapath(center_node_type,center_node_index)

        if not np.any(patient_neg_sample == None):
            single_patient_sample = np.concatenate((patient_sample_pos,patient_neg_sample),axis=0)
        else:
            single_patient_sample = patient_sample_pos
        if not np.any(item_neg_sample == None):
            single_item_sample = np.concatenate((item_sample_pos,item_neg_sample),axis=0)
        else:
            single_item_sample = item_sample_pos
        if not np.any(diag_neg_sample == None):
            single_diag_sample = np.concatenate((diag_sample_pos,diag_neg_sample),axis=0)
        else:
            single_diag_sample = diag_sample_pos

        return single_patient_sample, single_item_sample, single_diag_sample

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

        #self.length_patient_pos = len(patient_nodes)
        #self.length_item_pos = len(item_nodes)
        #self.length_diag_pos = len(diag_nodes)

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

        self.length_patient_neg = len(neg_nodes_patient)
        self.length_item_neg = len(neg_nodes_item)
        self.length_diag_neg = len(neg_nodes_diag)

        if not neg_nodes_patient == []:
            index = 0
            patient_neg_sample = np.zeros((self.length_patient_neg, self.item_size))
            for i in neg_nodes_patient:
                one_sample_neg_patient = self.assign_value_patient(i)
                patient_neg_sample[index,:] = one_sample_neg_patient
                index += 1

        if not neg_nodes_item == []:
            index = 0
            item_neg_sample = np.zeros((self.length_item_neg, self.item_size))
            for i in neg_nodes_item:
                one_sample_neg_item = self.assign_value_item(i)
                item_neg_sample[index,:] = one_sample_neg_item
                index += 1

        if not neg_nodes_diag == []:
            index = 0
            diag_neg_sample = np.zeros((self.length_diag_neg, self.diagnosis_size))
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
        metapath_whole = []
        for i in range(self.walk_length_iter):
            metapath_whole += meta_path_type

        self.metapath_whole = [i[0] for i in groupby(metapath_whole)]
        walk = self.generate_next_node(node_type, start_index, self.metapath_whole)

        return walk





