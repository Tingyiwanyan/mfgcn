import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby

class LSTM_model():
    """
    Create LSTM model for EHR data
    """
    def __init__(self,kg,hetro_model,data_process):
        """
        initialization for varies variables
        """
        self.kg = kg
        self.data_process = data_process
        self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.length_train = len(self.train_data)
        self.batch_size = 16
        self.time_sequence = 3
        self.latent_dim = 100
        self.latent_dim_cell_state = 100
        self.epoch = 10
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.input_seq = []
        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.placeholder(tf.float32, [None, self.latent_dim])
        self.input_x = tf.placeholder(tf.float32,[None,self.time_sequence,self.item_size])
        self.input_y_diag_single = tf.placeholder(tf.float32,[None,self.diagnosis_size])
        self.input_y_diag = tf.placeholder(tf.float32,[None,self.time_sequence,self.diagnosis_size])
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_softmax_convert = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
            tf.Variable(self.init_forget_gate(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.weight_softmax_convert = \
            tf.Variable(self.init_softmax_convert(shape=(self.latent_dim,self.diagnosis_size)))
        self.weight_output_gate = \
            tf.Variable(self.init_output_gate(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        self.bias_softmax_convert = tf.Variable(self.init_softmax_convert(shape=(self.diagnosis_size,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))


    def lstm_cell(self):
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
        self.lstm_cell()
        self.softmax_loss()
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()


    def get_batch_train(self,start_index):
        """
        get training batch data
        """
        self.train_one_batch = np.zeros((self.batch_size,self.time_sequence,self.item_size))
        self.logit_one_batch = np.zeros((self.batch_size,self.time_sequence,self.diagnosis_size))
        self.train_logit_single = np.zeros((self.batch_size,self.diagnosis_size))
        for i in range(self.batch_size):
            patient_id = self.train_data[start_index+i]
            time_series = self.kg.dic_patient_addmission[patient_id]['time_series'][0:self.time_sequence]
            index_time_sequence = 0
            for j in time_series:
                one_data = self.hetro_model.assign_value_patient(j)
                self.train_one_batch[i,index_time_sequence,:] = one_data
                one_data_logit = self.hetro_model.assign_multi_hot(j)
                self.logit_one_batch[i,index_time_sequence,:] = one_data_logit
                index_time_sequence += 1
            one_data_logit_single = self.hetro_model.assign_multi_hot(time_series[0])
            self.train_logit_single[i,:] = one_data_logit_single

    def train(self):
        """
        train the system
        """
        init_hidden_state = np.zeros((self.batch_size,self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train)/self.batch_size))
        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.get_batch_train(i+self.batch_size)
                self.err_ = self.sess.run([self.cross_entropy, self.train_step_cross_entropy,self.init_hiddenstate,self.output_layer,self.logit_softmax],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.input_y_diag: self.logit_one_batch,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_[0])

















