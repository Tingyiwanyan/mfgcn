import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
from process_data import Data_process
#import utils

class model_optimization(Data_process):
    """
    define decoding optimization model, for the use of all embedding methods
    """
    def __init__(self,data_set,option,option_lp_nc):
        """
        Define parameters
        """
        Data_process.__init__(self,data_set,option_lp_nc)
        #self.G = G
        self.batch_size = 64
        if data_set == 1:
            """
            Aminer data
            """
            self.attribute_size = 5
            self.walk_length = 10
            self.latent_dim = 100
            # latent_dim_second = 100
            self.latent_dim_gcn = 8
            self.latent_dim_gcn2 = 100
            self.latent_dim_a = 100
            self.negative_sample_size = 100
        if data_set == 2:
            """
            citeceer data set 
            """
            self.attribute_size = 3703
            self.walk_length = 10
            self.latent_dim = 100
            self.latent_dim_gcn = 500
            self.latent_dim_gcn2 = 300
            self.negative_sample_size = 100
            self.class_num = 6
            self.y_label = tf.placeholder(tf.float32, [None, self.class_num])
        self.data_length = len(list(self.G.nodes()))
        self.length = len(list(self.G.nodes()))
        self.sess = None
        self.total_loss = None
        self.Dense_layer_fc_gcn = None
        self.Dense4_n2v = None
        self.combined_embed = None
        self.option = option
        self.filter_num = 30


        """
        Input of GCN aggregator
        """
        self.x_gcn = tf.placeholder(tf.float32,
                                    [None, 1 + self.walk_length + self.negative_sample_size, self.attribute_size])
        self.x_skip = tf.placeholder(tf.float32, [None, self.walk_length, self.attribute_size])
        self.x_negative = tf.placeholder(tf.float32, [None, self.negative_sample_size, self.attribute_size])
        self.x_label = tf.placeholder(tf.float32, [None, self.data_length])

        """
        Input for n2v Structure only aggregator
        """
        self.x_n2v = tf.placeholder(tf.float32,
                                    [None, 1 + self.walk_length + self.negative_sample_size, self.length])
        self.x_skip_n2v = tf.placeholder(tf.float32, [None, self.walk_length, self.length])
        self.x_negative_n2v = tf.placeholder(tf.float32, [None, self.negative_sample_size, self.length])

        """
        Input of center node
        """
        self.x_center = tf.placeholder(tf.float32, [None, self.attribute_size])
        """
        Input of target vector
        """
        self.y_mean_pooling = tf.placeholder(tf.float32, [None, self.attribute_size])

        """
        Input of skip-gram vectors
        """
        self.z_skip_gram = tf.placeholder(tf.float32, [None, self.walk_length, self.latent_dim])

        self.z_skip_gram_normalize = tf.math.l2_normalize(self.z_skip_gram, axis=1)

        """
        Input negative sampling samples
        """
        self.z_negative_sampling = tf.placeholder(tf.float32, [None, self.negative_sample_size, self.latent_dim])

        self.z_negative_sampling_normalize = tf.math.l2_normalize(self.z_negative_sampling, axis=1)

        self.h1 = None
        self.h2 = None

        self.mse = None
        self.negative_sum = None

        self.x_origin = None
        self.Decoding_reduce = None

        #self.logits = None

    def build_first_layer(self):
        """
        Dense_gcn = tf.layers.dense(inputs=self.x_gcn,
                                    units=self.latent_dim_gcn,
                                    kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                    activation=tf.nn.relu)

        Dense_gcn2 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn3 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn4 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn5 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn6 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn7 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn8 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn9 = tf.layers.dense(inputs=self.x_gcn,
                                     units=self.latent_dim_gcn,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        Dense_gcn10 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn11 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn12 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn13 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn14 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn15 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn16 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn17 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn18 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn19 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense_gcn20 = tf.layers.dense(inputs=self.x_gcn,
                                      units=self.latent_dim_gcn,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)
        """


        """
        Lable for link prediction
        """
        # y_label = tf.placeholder(tf.float32,[None,walk_length+negative_sample_size])

        """
        Perform concatenation operation for posterior probability
        """
        # concat_posterior = tf.concat([x_center, Dense_gcn, y_mean_pooling],1)

        self.h1 = tf.concat([Dense_gcn, Dense_gcn2, Dense_gcn3,
                                      Dense_gcn4, Dense_gcn5, Dense_gcn6,
                                      Dense_gcn7, Dense_gcn8, Dense_gcn9,
                                      Dense_gcn10, Dense_gcn11, Dense_gcn12,
                                      Dense_gcn13, Dense_gcn14, Dense_gcn15,
                                      Dense_gcn16, Dense_gcn17, Dense_gcn18,
                                      Dense_gcn19, Dense_gcn20], 2)

        self.Dense_layer_fc_gcn = tf.layers.dense(inputs=self.h1,
                                                  units=self.latent_dim,
                                                  kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                  activation=tf.nn.elu,
                                                  name='embedding_gcn')

    def build_second_layer(self):

        Dense2_gcn1 = tf.layers.dense(inputs=self.h1,
                                      units=self.latent_dim_gcn2,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense2_gcn2 = tf.layers.dense(inputs=self.h1,
                                      units=self.latent_dim_gcn2,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense2_gcn3 = tf.layers.dense(inputs=self.h1,
                                      units=self.latent_dim_gcn2,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense2_gcn4 = tf.layers.dense(inputs=self.h1,
                                      units=self.latent_dim_gcn2,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        Dense2_gcn5 = tf.layers.dense(inputs=self.h1,
                                      units=self.latent_dim_gcn2,
                                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                      activation=tf.nn.relu)

        concat_posterior2 = tf.concat([Dense2_gcn1, Dense2_gcn2, Dense2_gcn3,
                                       Dense2_gcn4, Dense2_gcn5], 2)
        """
        We use elu to avoid too sparse data
        """
        self.Dense_layer_fc_gcn = tf.layers.dense(inputs=concat_posterior2,
                                             units=self.latent_dim,
                                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                             activation=tf.nn.elu,
                                             name='embedding_gcn')

    def n2v(self):

        Dense3_n2v = tf.layers.dense(inputs=self.x_n2v,
                                     units=1024,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.relu)

        self.Dense4_n2v = tf.layers.dense(inputs=Dense3_n2v,
                                     units=100,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.elu,
                                     name='embedding_n2v')

    def SGNN_loss(self):
        """
        Implement sgnn with new structure
        """
            #self.latent_dim = 2 * self.latent_dim

        #sum_doc_regularization = tf.reduce_sum(tf.reduce_sum(doc_regularization, axis=2), axis=1)

        #mean_sum_doc = tf.reduce_mean(sum_doc_regularization)
        """
        negative_training = tf.broadcast_to(tf.expand_dims(x_negative,1),[batch_size,walk_length,negative_sample_size,latent_dim])

        skip_training = tf.broadcast_to(tf.expand_dims(x_skip,2),[batch_size,walk_length,negative_sample_size,latent_dim])

        negative_training_norm = tf.math.l2_normalize(negative_training,axis=3)
        """

        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.negative_sample_size, self.latent_dim])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)
        """
        skip_mean = tf.reduce_mean(dot_prod_sum,1)

        log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(skip_mean)))

        sum_log_dot_prod = tf.reduce_sum(log_dot_prod,1)
        """
        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.walk_length, self.latent_dim])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))

        #regularized_negative_sum = tf.math.add(self.negative_sum, mean_sum_doc)

    def mse_loss(self):
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_layer_fc_gcn, idx_origin, axis=1)

        """
        mse loss
        """
        Decoding_auto_encoder = tf.layers.dense(inputs=self.x_origin,
                                                units=self.attribute_size,
                                                kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                activation=tf.nn.sigmoid)

        self.Decoding_reduce = tf.squeeze(Decoding_auto_encoder)
        # mse = tf.losses.mean_squared_error(y_mean_pooling, Decoding_reduce)
        self.mse = tf.losses.mean_squared_error(self.x_center, self.Decoding_reduce)

    def supervised_loss(self):
        """
        supervised loss
        """
        #idx_origin = tf.constant([0])
        #self.x_origin = tf.gather(self.Dense_layer_fc_gcn, idx_origin, axis=1)

        """
        supervised loss
        """
        self.logit_decode = tf.layers.dense(inputs=self.x_origin,
                                        units=self.class_num,
                                        kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                        activation=tf.nn.sigmoid)
        self.logit_softmax = tf.nn.softmax(self.logit_decode)
        self.logit_softmax_reduce = tf.squeeze(self.logit_softmax)
        #self.loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_label,logits=self.logits))
        #self.loss_sup = tf.losses.mean_squared_error(self.y_label, self.logit_softmax_reduce)
        self.cross_entropy = tf.reduce_mean(tf.math.negative(tf.reduce_sum(tf.math.multiply(self.y_label,tf.log(self.logit_softmax_reduce)), reduction_indices=[1])))

    def pick_model(self):
        idx_origin = tf.constant([0])
        idx_skip = tf.constant([i + 1 for i in range(self.walk_length)])
        idx_negative = tf.constant([i + 1 + self.walk_length for i in range(self.negative_sample_size)])

        # x_origin = tf.gather(x_gcn,idx_origin,axis=1)
        # x_skip = tf.gather(x_gcn,idx_skip,axis=1)
        # x_negative = tf.gather(x_gcn,idx_negative,axis=1)

        # x_output = tf.squeeze(x_origin)
        if self.option == 1:
            # doc_regularization = tf.multiply(self.Dense_layer_fc_gcn, self.Dense_layer_fc_gcn)
            self.x_origin = tf.gather(self.Dense_layer_fc_gcn, idx_origin, axis=1)
            self.x_skip = tf.gather(self.Dense_layer_fc_gcn, idx_skip, axis=1)
            self.x_negative = tf.gather(self.Dense_layer_fc_gcn, idx_negative, axis=1)
        if self.option == 2:
            # doc_regularization = tf.multiply(self.Dense4_n2v, self.Dense4_n2v)
            self.x_origin = tf.gather(self.Dense4_n2v, idx_origin, axis=1)
            self.x_skip = tf.gather(self.Dense4_n2v, idx_skip, axis=1)
            self.x_negative = tf.gather(self.Dense4_n2v, idx_negative, axis=1)

        if self.option == 3:
            """
            combine structure and feature
            """
            """
            self.x_origin_gcn = tf.gather(self.Dense_layer_fc_gcn, idx_origin, axis=1)
            self.x_skip_gcn = tf.gather(self.Dense_layer_fc_gcn, idx_skip, axis=1)
            self.x_negative_gcn = tf.gather(self.Dense_layer_fc_gcn, idx_negative, axis=1)

            self.x_origin_n2v = tf.gather(self.Dense4_n2v, idx_origin, axis=1)
            self.x_skip_n2v = tf.gather(self.Dense4_n2v, idx_skip, axis=1)
            self.x_negative_n2v = tf.gather(self.Dense4_n2v, idx_negative, axis=1)

            self.x_origin = tf.concat([self.x_origin_gcn, self.x_origin_n2v], axis=2)
            self.x_skip = tf.concat([self.x_skip_gcn, self.x_skip_n2v], axis=2)
            self.x_negative = tf.concat([self.x_negative_gcn, self.x_negative_n2v], axis=2)
            """

            self.Dense_concat = tf.concat([self.Dense_layer_fc_gcn, self.Dense4_n2v], axis=2)

            self.Dense_combine = tf.layers.dense(inputs=self.Dense_concat,
                                                 units=100,
                                                 kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                                 activation=tf.nn.elu,
                                                 name='embedding_concat')

            self.x_origin = tf.gather(self.Dense_combine, idx_origin, axis=1)
            self.x_skip = tf.gather(self.Dense_combine, idx_skip, axis=1)
            self.x_negative = tf.gather(self.Dense_combine, idx_negative, axis=1)

    def config_model(self):
        if self.option == 1:
            self.build_first_layer()
            self.build_second_layer()
            #self.mse_loss()

        if self.option == 2:
            self.n2v()
            #self.supervised_loss()

        if self.option == 3:
            self.build_first_layer()
            #self.build_second_layer()
            self.n2v()
            #self.supervised_loss()

        self.pick_model()
        self.SGNN_loss()
        self.supervised_loss()

        #self.total_loss = tf.math.add(self.loss_sup, self.negative_sum)

        #self.train_step_auto = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.train_step_neg = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()






