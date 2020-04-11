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
    def __init__(self,kg,hetro_model):
        """
        initialization for varies variables
        """
        self.kg = kg
        self.batch_size = 32
        self.hetro_model = hetro_model
        self.time_sequence = 3
        self.latent_dim = 100
        self.latent_dim_cell_state = 100
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.placeholder(tf.float32, [None, self.latent_dim])
        self.input_x = tf.placeholder(tf.float32,[None,self.time_sequence,self.item_size])
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



    def LSTM_cell(self):
        """
        Implementation of LSTM cell for recursive usage
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
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur,self.weight_forget_gate),self.bias_forget_gate))
            info_cur = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur,self.weight_info_gate),self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur,self.weight_cell_state),self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i ==0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state,info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur,self.weight_output_gate),self.bias_output_gate))
            hidden_current = tf.multiply(output_gate,cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i],1)
        hidden_rep = tf.concat(hidden_rep,1)
        self.check = concat_cur

        return hidden_rep

    def softmax_loss(self,hidden_rep):
        """
        Implement softmax loss layer
        """
        self.output_layer = tf.layers.dense(inputs=hidden_rep,
                                           units=self.diagnosis_size,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        self.logit_softmax = tf.nn.softmax(self.output_layer)
        self.cross_entropy = \
            tf.reduce_mean(
            tf.math.negative(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.math.multiply(
                            LSTM_model.input_y_diag,tf.log(
                                LSTM_model.logit_softmax)),reduction_indices=[1]),reduction_indices=[1])))

    def config_model(self):
        """
        Model configuration
        """
        self.LSTM_cell()
        hidden_rep = self.LSTM_cell()
        self.softmax_loss(hidden_rep)
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    #def get










