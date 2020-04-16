import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby

class hetero_model_modify():
    """
    Create deep heterogenous embedding model, by using metapath learning and TransE model.
    """

    def __init__(self, kg, data_process):
        self.data_process = data_process
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_hadm_id
        self.test_data_lstm = self.data_process.test_patient
        self.length_test = len(self.test_data)
        self.length_train = len(self.train_data)
        #self.length_train = len(self.train_data)
        self.time_sequence = 3
        if self.time_sequence == 1:
            self.train_data_sgnn = self.data_process.train_hadm_id
        else:
            self.train_data_sgnn = kg.dic_patient.keys()

        self.length_train_hadm = len(self.train_data_sgnn)
        self.neg_time_length = 15
        self.batch_size = 16
        self.epoch = 6
        self.latent_dim = 300
        self.latent_dim_lstm = 100
        self.negative_sample_each_type = 50
        self.positive_sample_each_type = 10
        self.positive_sample_size = self.positive_sample_each_type*3
        self.negative_sample_size = self.negative_sample_each_type*2
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.kg = kg
        self.x_origin = None
        self.x_negative = None

        self.shape_relation = (self.latent_dim,)
        self.init_test = tf.keras.initializers.he_normal(seed=None)
        self.init_diag = tf.keras.initializers.he_normal(seed=None)
        self.init_time = tf.keras.initializers.he_normal(seed=None)
        """
        initial relation type test
        """
        self.relation_test = tf.Variable(self.init_test(shape=self.shape_relation))

        """
        initial relation type diag
        """
        self.relation_diag = tf.Variable(self.init_diag(shape=self.shape_relation))

        """
        initial relation type time
        """
        self.relation_time = tf.Variable(self.init_time(shape=self.shape_relation))

        """
        define test vector
        """
        self.item = tf.placeholder(
            tf.float32, [None,self.positive_sample_each_type+self.negative_sample_each_type,self.item_size])

        """
        define patient vector initial visit
        """
        self.patient = tf.placeholder(tf.float32, [None, 1+self.positive_sample_each_type,self.item_size])

        """
        define diagnosis vector
        """
        self.diagnosis = tf.placeholder(
            tf.float32, [None, self.positive_sample_each_type+self.negative_sample_each_type,self.diagnosis_size])

        """
        define patient time series
        """
        self.patient_time = tf.placeholder(
            tf.float32, [None, self.time_sequence+self.neg_time_length,self.item_size])

        """
        define patient weight vector
        """
        self.init_weight_patient = tf.keras.initializers.he_normal(seed=None)
        self.weight_patient = \
            tf.Variable(self.init_weight_patient(shape=(self.item_size, self.latent_dim)))
        self.bias_patient = tf.Variable(self.init_weight_patient(shape=(self.latent_dim,)))

        """
        define item weight vector
        """
        self.init_weight_item = tf.keras.initializers.he_normal(seed=None)
        self.weight_item = \
            tf.Variable(self.init_weight_item(shape=(self.item_size, self.latent_dim)))
        self.bias_item = tf.Variable(self.init_weight_item(shape=(self.latent_dim,)))

        """
        define diagnosis weight vector
        """
        self.init_weight_diag = tf.keras.initializers.he_normal(seed=None)
        self.weight_diag = \
            tf.Variable(self.init_weight_diag(shape=(self.diagnosis_size, self.latent_dim)))
        self.bias_diag = tf.Variable(self.init_weight_diag(shape=(self.latent_dim,)))

        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.placeholder(tf.float32, [None, self.latent_dim_lstm])
        self.init_hiddenstate_patient = tf.placeholder(tf.float32,[None,1+self.positive_sample_each_type,self.latent_dim_lstm])
        #self.input_x = tf.placeholder(tf.float32, [None, self.time_sequence, self.item_size])
        self.input_x = tf.placeholder(tf.float32, [None, self.time_sequence, self.latent_dim])
        self.input_y_diag_single = tf.placeholder(tf.float32, [None, self.diagnosis_size])
        self.input_y_diag = tf.placeholder(tf.float32, [None, self.time_sequence, self.diagnosis_size])
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_softmax_convert = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
            tf.Variable(self.init_forget_gate(shape=(self.latent_dim + self.latent_dim_lstm, self.latent_dim_lstm)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.latent_dim + self.latent_dim_lstm, self.latent_dim_lstm)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.latent_dim + self.latent_dim_lstm, self.latent_dim_lstm)))
        self.weight_softmax_convert = \
            tf.Variable(self.init_softmax_convert(shape=(self.latent_dim_lstm, self.diagnosis_size)))
        self.weight_output_gate = \
            tf.Variable(self.init_output_gate(shape=(self.latent_dim + self.latent_dim_lstm, self.latent_dim_lstm)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim_lstm,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim_lstm,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim_lstm,)))
        self.bias_softmax_convert = tf.Variable(self.init_softmax_convert(shape=(self.diagnosis_size,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim_lstm,)))

    def lstm_cell_time(self):
        """
        define lstm cell for time series
        """
        cell_state = []
        hidden_rep = []
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.input_x, i, axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate,x_input_cur],1)
            else:
                concat_cur = tf.concat([hidden_rep[i-1],x_input_cur],1)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_forget_gate),self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_info_gate),self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur,self.weight_cell_state),self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i ==0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state,info_cell_state)
            output_gate = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_output_gate),self.bias_output_gate))
            hidden_current = tf.multiply(output_gate,cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i],1)
        self.hidden_rep = tf.concat(hidden_rep,1)
        self.check = concat_cur

    def lstm_single_cell(self):
        """
        build lstm for single cell
        """
        concat_cur = tf.concat([self.init_hiddenstate_patient, self.patient], 2)
        forget_cur = \
            tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_forget_gate), self.bias_forget_gate))
        info_cur = \
            tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_info_gate), self.bias_info_gate))
        cellstate_cur = \
            tf.math.tanh(tf.math.add(tf.matmul(concat_cur, self.weight_cell_state), self.bias_cell_state))
        info_cell_state = tf.multiply(info_cur, cellstate_cur)
        output_gate = \
            tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur, self.weight_output_gate), self.bias_output_gate))
        self.Dense_patient = tf.multiply(output_gate, cellstate_cur)

    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        self.output_layer = tf.layers.dense(inputs=self.hidden_rep,
                                           units=self.diagnosis_size,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        self.logit_softmax = tf.nn.softmax(self.output_layer)
        #self.cross_entropy = tf.reduce_mean(tf.math.negative(
        #    tf.reduce_sum(tf.math.multiply(self.input_y_diag_single, tf.log(self.logit_softmax)), reduction_indices=[1])))

        self.cross_entropy = \
            tf.reduce_mean(
            tf.math.negative(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.math.multiply(
                            self.input_y_diag,tf.log(
                                self.logit_softmax)),reduction_indices=[1]),reduction_indices=[1])))


    def config_model(self):
        """
        Model configuration
        """
        self.lstm_cell_time()
        #self.lstm_single_cell()
        self.softmax_loss()
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.build_hetero_model()
        self.get_latent_rep_hetero()
        self.SGNN_loss()
        self.train_step_neg = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def build_hetero_model(self):
        """
        build heterogenous graph learning model
        """

        """
        build item projection layer
        """
        self.Dense_item_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.item,self.weight_item),self.bias_item))

        """
        build diagnosis projection layer
        """
        self.Dense_diag_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.diagnosis, self.weight_diag), self.bias_diag))

        """
        build patient projection layer
        """
        self.Dense_patient = \
            tf.nn.relu(tf.math.add(tf.matmul(self.patient, self.weight_patient), self.bias_patient))

        self.Dense_item = tf.math.add(self.Dense_item_, self.relation_test)
        self.Dense_diag = tf.math.subtract(self.Dense_diag_, self.relation_diag)

        """
        build patient time series projection layer
        """
       # self.Dense_patient_time = \
         #   tf.nn.relu(tf.math.add(tf.matmul(self.patient_time, self.weight_patient), self.bias_patient))

       # self.Dense_patient_time = tf.math.subtract(self.Dense_patient_time,self.relation_time)


    def get_latent_rep_hetero(self):
        """
        get latent representation, center node is patient
        """
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_patient, idx_origin, axis=1)
        patient_idx_skip = tf.constant([i + 1 for i in range(self.positive_sample_each_type)])
        self.x_skip_patient = tf.gather(self.Dense_patient, patient_idx_skip, axis=1)
        """
        getting positive samples
        """
        item_idx_skip = tf.constant([i for i in range(self.positive_sample_each_type)])
        self.x_skip_item = tf.gather(self.Dense_item, item_idx_skip, axis=1)
        diag_idx_skip = tf.constant([i for i in range(self.positive_sample_each_type)])
        self.x_skip_diag = tf.gather(self.Dense_diag, diag_idx_skip, axis=1)
        """
        getting negative samples
        """
        item_idx_negative = \
            tf.constant([i + self.positive_sample_each_type for i in range(self.negative_sample_each_type)])
        self.x_negative_item = tf.gather(self.Dense_item, item_idx_negative, axis=1)
        diag_idx_negative = \
            tf.constant([i + self.positive_sample_each_type for i in range(self.negative_sample_each_type)])
        self.x_negative_diag = tf.gather(self.Dense_diag, diag_idx_negative, axis=1)

        """
        combine skip samples and negative samples
        """
        self.x_skip = self.x_skip_patient
        self.x_skip = tf.concat([self.x_skip, self.x_skip_item], axis=1)
        self.x_skip = tf.concat([self.x_skip, self.x_skip_diag], axis=1)

        """
        prepare negative data
        """
        self.x_negative = self.x_negative_item
        self.x_negative = tf.concat([self.x_negative, self.x_negative_diag], axis=1)

    def SGNN_loss(self):
        """
        implement sgnn loss
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, self.negative_sample_size, self.latent_dim])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.positive_sample_size, self.latent_dim])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))

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
                one_sample[index] = (np.float(ave_value) - mean) / std

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

    def assign_value_diag(self, diagid):
        one_sample = np.zeros(self.diagnosis_size)
        index = self.kg.dic_diag[diagid]['diag_index']
        one_sample[index] = 1

        return one_sample

    """
    assign multi-hot diagnosis to one patient
    """

    def assign_multi_hot(self, patientid):
        one_sample = np.zeros(self.diagnosis_size)
        for i in self.kg.dic_patient[patientid]['neighbor_diag']:
            index = self.kg.dic_diag[i]['diag_index']
            one_sample[index] = 1

        return one_sample


    """
    prepare data positive sampling
    """
    def get_positive_samples(self,center_node_index):
        self.pos_nodes_item = []
        self.pos_nodes_diag = []
        self.pos_nodes_patient = []
        self.diag_pos_index_samples = []
        """
        get pos set for diag
        """
        diag_neighbor_nodes = self.kg.dic_patient[center_node_index]['neighbor_diag']
        for j in range(self.positive_sample_each_type):
            index_sample = np.int(np.floor(np.random.uniform(0, len(diag_neighbor_nodes), 1)))
            self.pos_nodes_diag.append(diag_neighbor_nodes[index_sample])
            self.diag_pos_index_samples.append(diag_neighbor_nodes[index_sample])
        """
        get pos set for patient
        """
        for j in self.diag_pos_index_samples:
            patient_neighbor_nodes = self.kg.dic_diag[j]['neighbor_patient']
            index_sample = np.int(np.floor(np.random.uniform(0, len(patient_neighbor_nodes), 1)))
            self.pos_nodes_patient.append(patient_neighbor_nodes[index_sample])
        """
        get pos set for item
        """
        item_neighbor_nodes = self.kg.dic_patient[center_node_index]['itemid'].keys()
        for j in range(self.positive_sample_each_type):
            index_sample = np.int(np.floor(np.random.uniform(0, len(item_neighbor_nodes), 1)))
            self.pos_nodes_item.append(item_neighbor_nodes[index_sample])

    """
    get positive nodes expression
    """
    def get_positive_sample_metapath(self):
        self.diag_pos_sample = np.zeros((self.positive_sample_each_type,self.diagnosis_size))
        index = 0
        for i in self.pos_nodes_diag:
            one_sample_pos_diag = self.assign_value_diag(i)
            self.diag_pos_sample[index,:] = one_sample_pos_diag
            index += 1

        self.item_pos_sample = np.zeros((self.positive_sample_each_type,self.item_size))
        index = 0
        for i in self.pos_nodes_item:
            one_sample_pos_item = self.assign_value_item(i)
            self.item_pos_sample[index,:] = one_sample_pos_item
            index += 1
        self.patient_pos_sample = np.zeros((self.positive_sample_each_type,self.item_size))
        index = 0
        for i in self.pos_nodes_patient:
            one_sample_pos_patient = self.assign_value_patient(i)
            self.patient_pos_sample[index,:] = one_sample_pos_patient
            index += 1



    """
    prepare data for one metapath negative sample
    """

    def get_negative_sample_metapath(self):

        self.diag_neg_sample = np.zeros((self.negative_sample_each_type, self.diagnosis_size))
        index = 0
        for i in self.neg_nodes_diag:
            one_sample_neg_diag = self.assign_value_diag(i)
            self.diag_neg_sample[index, :] = one_sample_neg_diag
            index += 1

        self.item_neg_sample = np.zeros((self.negative_sample_each_type,self.item_size))
        index = 0
        for i in self.neg_nodes_item:
            one_sample_neg_item = self.assign_value_item(i)
            self.item_neg_sample[index, :] = one_sample_neg_item
            index += 1

    """
    prepare data for negative hererogenous sampling
    """

    def get_negative_samples(self, center_node_index):
        self.neg_nodes_item = []
        self.neg_nodes_diag = []
        """
        get neg set for diag
        """
        diag_neighbor_nodes = self.kg.dic_patient[center_node_index]['neighbor_diag']
        whole_diag_nodes = self.kg.dic_diag.keys()
        #gene_neighbor_nodes = gene_neighbor_nodes + self.walk_gene
        neg_set_diag = [i for i in whole_diag_nodes if i not in diag_neighbor_nodes]
        for j in range(self.negative_sample_each_type):
            index_sample = np.int(np.floor(np.random.uniform(0, len(neg_set_diag), 1)))
            self.neg_nodes_diag.append(neg_set_diag[index_sample])
        """
        get neg set for item
        """
        item_neighbor_nodes = self.kg.dic_patient[center_node_index]['itemid'].keys()
        whole_item_nodes = self.kg.dic_item.keys()
        neg_set_item = [i for i in whole_item_nodes if i not in item_neighbor_nodes]
        for j in range(self.negative_sample_each_type):
            index_sample = np.int(np.floor(np.random.uniform(0, len(neg_set_item), 1)))
            self.neg_nodes_item.append(neg_set_item[index_sample])

    def get_one_batch_sgnn(self, start_index):
        """
        get train data for sgnn model
        """
        self.patient_sample = np.zeros((self.batch_size,1+self.positive_sample_each_type,self.item_size))
        self.diag_sample = \
            np.zeros((self.batch_size,self.positive_sample_each_type+self.negative_sample_each_type,self.diagnosis_size))
        self.item_sample = \
            np.zeros((self.batch_size,self.positive_sample_each_type+self.negative_sample_each_type,self.item_size))
        for i in range(self.batch_size):
            center_node_index = self.train_data_sgnn[i+start_index]
            center_patient_node = np.expand_dims(self.assign_value_patient(center_node_index),axis=0)
            self.get_positive_samples(center_node_index)
            self.get_positive_sample_metapath()
            self.get_negative_samples(center_node_index)
            self.get_negative_sample_metapath()
            single_patient = np.concatenate((center_patient_node,self.patient_pos_sample))
            single_diag = np.concatenate((self.diag_pos_sample,self.diag_neg_sample))
            single_item = np.concatenate((self.item_pos_sample,self.item_neg_sample))
            self.patient_sample[i,:,:] = single_patient
            self.diag_sample[i,:,:] = single_diag
            self.item_sample[i,:,:] = single_item


    def get_batch_train(self, start_index):
        """
        get training batch data for lstm model
        """
        self.one_data_embedded = np.zeros((1,1+self.positive_sample_each_type,self.item_size))
        self.train_one_batch = np.zeros((self.batch_size, self.time_sequence, self.latent_dim))
        self.logit_one_batch = np.zeros((self.batch_size, self.time_sequence, self.diagnosis_size))
        self.train_logit_single = np.zeros((self.batch_size, self.diagnosis_size))
        for i in range(self.batch_size):
            patient_id = self.train_data[start_index + i]
            time_series = self.kg.dic_patient_addmission[patient_id]['time_series'][0:self.time_sequence]
            index_time_sequence = 0
            for j in time_series:
                one_data = self.assign_value_patient(j)
                self.one_data_embedded[0,0,:] = one_data
                self.embed = self.sess.run(self.Dense_patient,feed_dict={self.patient:self.one_data_embedded})
                self.train_one_batch[i, index_time_sequence, :] = self.embed[0][0]
                one_data_logit = self.assign_multi_hot(j)
                self.logit_one_batch[i, index_time_sequence, :] = one_data_logit
                index_time_sequence += 1
            one_data_logit_single = self.assign_multi_hot(time_series[0])
            self.train_logit_single[i, :] = one_data_logit_single

    def train(self):
        """
        train the system
        """
        init_hidden_state = np.zeros((self.batch_size, self.latent_dim_lstm))
        iteration = np.int(np.floor(np.float(self.length_train) / self.batch_size))
        init_hidden_state_patient = np.zeros((self.batch_size, 1 + self.positive_sample_each_type, self.latent_dim))
        iteration_sgnn = np.int(np.floor(np.float(self.length_train_hadm) / self.batch_size))
        if not self.time_sequence == 1:
            epoch = 3
        else:
            epoch = self.epoch
        for kk in range(epoch):
            for k in range(iteration_sgnn):
                self.get_one_batch_sgnn(k + self.batch_size)
                err_ = self.sess.run([self.negative_sum, self.train_step_neg], feed_dict={self.patient: self.patient_sample,
                                                                                          self.diagnosis: self.diag_sample,
                                                                                          self.item: self.item_sample})
                                                                                          #self.init_hiddenstate_patient: init_hidden_state_patient})
                print(err_[0])

        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.get_batch_train(i + self.batch_size)
                self.err_ = self.sess.run(
                    [self.cross_entropy, self.train_step_cross_entropy, self.init_hiddenstate, self.output_layer,
                     self.logit_softmax],
                    feed_dict={self.input_x: self.train_one_batch,
                               self.input_y_diag: self.logit_one_batch,
                               self.init_hiddenstate: init_hidden_state})
                print(self.err_[0])

    def test_lstm(self):
        """
        return test f1 score
        """
        length_test_patient = len(self.test_data_lstm)
        self.one_data_embedded = np.zeros((1, 1 + self.positive_sample_each_type, self.item_size))
        self.test_output = np.zeros((length_test_patient,self.time_sequence,self.latent_dim))
        self.test_logit_data = np.zeros((length_test_patient,self.time_sequence,self.diagnosis_size))
        init_hidden_state = np.zeros((length_test_patient, self.latent_dim_lstm))
        for i in range(length_test_patient):
            patient_id = self.test_data_lstm[i]
            time_series = self.kg.dic_patient_addmission[patient_id]['time_series'][0:self.time_sequence]
            index_time_sequence = 0
            for j in time_series:
                one_data = self.assign_value_patient(j)
                self.one_data_embedded[0, 0, :] = one_data
                self.embed = self.sess.run(self.Dense_patient, feed_dict={self.patient: self.one_data_embedded})
                self.test_output[i, index_time_sequence, :] = self.embed[0][0]
                one_data_logit = self.assign_multi_hot(j)
                self.test_logit_data[i, index_time_sequence, :] = one_data_logit
                index_time_sequence += 1
        self.logit_output = self.sess.run(self.logit_softmax, feed_dict={self.input_x: self.test_output,
                                                                         self.init_hiddenstate: init_hidden_state})

        self.rate_total_1 = []
        self.rate_total_2 = []
        self.rate_total_3 = []
        for k in range(length_test_patient):
            test_sample = self.logit_output[k, 0, :]
            actual_logit = self.test_logit_data[k, 0, :]
            detect = np.where(test_sample > 0.001)
            actual = np.where(actual_logit > 0.1)
            correct_detect = len([i for i in detect[0] if i in actual[0]])
            rate = float(correct_detect) / len(actual[0])
            self.rate_total_1.append(rate)
            test_sample = self.logit_output[k, 1, :]
            actual_logit = self.test_logit_data[k, 1, :]
            detect = np.where(test_sample > 0.001)
            actual = np.where(actual_logit > 0.1)
            correct_detect = len([i for i in detect[0] if i in actual[0]])
            rate = float(correct_detect) / len(actual[0])
            self.rate_total_2.append(rate)
            test_sample = self.logit_output[k, 2, :]
            actual_logit = self.test_logit_data[k, 2, :]
            detect = np.where(test_sample > 0.001)
            actual = np.where(actual_logit > 0.1)
            correct_detect = len([i for i in detect[0] if i in actual[0]])
            rate = float(correct_detect) / len(actual[0])
            self.rate_total_3.append(rate)

        self.f1_test_1 = np.mean(self.rate_total_1)
        self.f1_test_2 = np.mean(self.rate_total_2)
        self.f1_test_3 = np.mean(self.rate_total_3)

    """
    test model accuracy
    """

    def test(self):
        #self.patient_id_test = patient_id
        patient = np.zeros((self.length_test,1+self.positive_sample_each_type, self.item_size))
        index_p = 0
        for i in self.test_data:
            patient[index_p,0, :] = self.assign_value_patient(i)
            index_p += 1
        embed_patient = self.sess.run([self.Dense_patient], feed_dict={self.patient: patient})
        embed_patient_norm = np.zeros((self.length_test, self.latent_dim))
        for i in range(self.length_test):
            embed_patient_norm[i,:] = embed_patient[0][i][0] / np.linalg.norm(embed_patient[0][i][0])
        self.embed_item = np.zeros((len(self.kg.dic_item), self.latent_dim))
        self.embed_diag = np.zeros((len(self.kg.dic_diag), self.latent_dim))
        self.pos_score_item = np.zeros((self.length_test,len(self.kg.dic_item.keys())))
        self.pos_score_diag = np.zeros((self.length_test,len(self.kg.dic_diag.keys())))
        #for j in range(self.length_test):
        for i in self.kg.dic_item.keys():
            index = self.kg.dic_item[i]['item_index']
            single_item = np.zeros((1, self.positive_sample_each_type+self.negative_sample_each_type, self.item_size))
            single_item[0, 0,:] = self.assign_value_item(i)
            embed_single_item = self.sess.run([self.Dense_item], feed_dict={self.item: single_item})
            embed_item_single = embed_single_item[0][0][0] / np.linalg.norm(embed_single_item[0][0][0])
            self.embed_item[index, :] = embed_item_single
            #self.pos_score_item[j,index] = np.sum(np.multiply(embed_patient_norm[j,:], embed_item_single))

        self.pos_score_item = np.matmul(embed_patient_norm,self.embed_item.T)

        #for j in range(self.length_test):
        for i in self.kg.dic_diag.keys():
            index = self.kg.dic_diag[i]['diag_index']
            single_diag = np.zeros((1, self.positive_sample_each_type+self.negative_sample_each_type,self.diagnosis_size))
            single_diag[0,0, :] = self.assign_value_diag(i)
            embed_single_diag = self.sess.run([self.Dense_diag], feed_dict={self.diagnosis: single_diag})
            embed_diag_single = embed_single_diag[0][0][0] / np.linalg.norm(embed_single_diag[0][0][0])
            self.embed_diag[index, :] = embed_diag_single

        self.pos_score_diag = np.matmul(embed_patient_norm,self.embed_diag.T)
        """
        self.seq_diag = sorted(self.pos_score_diag)
        self.seq_item = sorted(self.pos_score_item)
        self.seq_diag.reverse()
        self.seq_item.reverse()

        self.index_diag = [list(self.pos_score_diag).index(v) for v in self.seq_diag]
        self.index_item = [list(self.pos_score_item).index(v) for v in self.seq_item]
        """

        self.correct_rate_item = np.zeros(self.length_test)
        for i in range(self.length_test):
            patient_id = self.test_data[i]
            correct_detect_num_item = 0
            num_item = len(self.kg.dic_patient[patient_id]['itemid'].keys())
            detect_index_item = np.array(np.where(self.pos_score_item[i,:] > 0.0))
            num_total_detect_item = len(detect_index_item)
            for j in self.kg.dic_patient[patient_id]['itemid'].keys():
                index_item = self.kg.dic_item[j]['item_index']
                if index_item in detect_index_item:
                    correct_detect_num_item += 1

            pos_rate_item = correct_detect_num_item / np.float(num_item)
            self.correct_rate_item[i] = pos_rate_item

        self.correct_rate_diag = np.zeros(self.length_test)
        for i in range(self.length_test):
            patient_id = self.test_data[i]
            correct_detect_num_diag = 0
            num_diag = len(self.kg.dic_patient[patient_id]['neighbor_diag'])
            detect_index_diag = np.array(np.where(self.pos_score_diag[i,:] > 0.0))
            num_total_detect_diag = len(detect_index_diag)
            for j in self.kg.dic_patient[patient_id]['neighbor_diag']:
                index_diag = self.kg.dic_diag[j]['diag_index']
                if index_diag in detect_index_diag:
                    correct_detect_num_diag += 1

            pos_rate_diag = correct_detect_num_diag / np.float(num_diag)
            self.correct_rate_diag[i] = pos_rate_diag

        #return pos_rate_item, pos_rate_diag

    """
    get the ranked item, diag for one patient
    """

    def recommandation(self):
        length_diag = len(np.where(np.array(self.seq_diag) > 0.9)[0])
        index_pick_diag = self.index_diag[0:length_diag]
        self.ICD = []
        for i in self.kg.dic_diag.keys():
            if self.kg.dic_diag[i]['diag_index'] in index_pick_diag:
                self.ICD.append(i)
        length_item = len(np.where(np.array(self.seq_item) > 0.5)[0])
        self.item_test = []
        index_pick_item = self.index_item[0:length_item]
        for i in self.kg.dic_item.keys():
            if self.kg.dic_item[i]['item_index'] in index_pick_item:
                self.item_test.append(i)

        self.diagnosis_recom = []
        self.item_test_recom = []
        for i in range(len(self.ICD)):
            ICD_diag = self.ICD[i]
            if ICD_diag in self.kg.diag_d_ar[:, 1]:
                index_ICD_diag = np.where(self.kg.diag_d_ar[:, 1] == ICD_diag)[0][0]
                diagnosis = self.kg.diag_d_ar[index_ICD_diag][3]
                self.diagnosis_recom.append([ICD_diag, diagnosis])
        for i in range(len(self.item_test)):
            item_id = self.item_test[i]
            index_item_id = np.where(self.kg.d_item_ar[:, 1] == item_id)[0][0]
            item_test_name = self.kg.d_item_ar[index_item_id][2]
            if item_id in self.kg.dic_patient[self.patient_id_test]['itemid'].keys():
                item_test_value = np.mean(self.kg.dic_patient[self.patient_id_test]['itemid'][item_id])
                self.item_test_recom.append([item_id, item_test_name, item_test_value])
