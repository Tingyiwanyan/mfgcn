import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby
from evaluation import cal_auc

class NN_model():
    """
    Create shalow neural network model for EHR data
    """
    def __init__(self,kg,hetro_model,data_process):
        self.kg = kg
        self.data_process = data_process
        self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_train_hadm = len(data_process.train_hadm_id)
        self.batch_size = 16
        self.latent_dim = 500
        self.epoch = 6
        self.resolution = 0.0001
        self.threshold_diag = 0.06
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.input_seq = []
        """
        define shallow neural network
        """
        self.input_x = tf.placeholder(tf.float32, [None, self.item_size])
        self.input_y_diag = tf.placeholder(tf.float32, [None, self.diagnosis_size])

    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        self.hidden_rep = tf.layers.dense(inputs=self.input_x,
                                          units=self.latent_dim,
                                          kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                          activation=tf.nn.relu)
        self.batch_normed = tf.keras.layers.BatchNormalization()
        self.hidden_batch_normed = self.batch_normed(self.hidden_rep)
        self.output_layer = tf.layers.dense(inputs=self.hidden_rep,
                                           units=self.diagnosis_size,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.elu)
        self.logit_softmax = tf.nn.softmax(self.output_layer)
        self.cross_entropy = tf.reduce_mean(tf.math.negative(
            tf.reduce_sum(tf.math.multiply(self.input_y_diag, tf.log(self.logit_softmax)), reduction_indices=[1])))

    def config_model(self):
        """
        Model configuration
        """
        self.softmax_loss()
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def get_batch_train(self,start_index):
        """
        get training batch data
        """
        self.train_one_batch = np.zeros((self.batch_size,self.item_size))
        self.logit_one_batch = np.zeros((self.batch_size,self.diagnosis_size))
        for i in range(self.batch_size):
            hadm_id = self.data_process.train_hadm_id[start_index+i]
            one_data = self.hetro_model.assign_value_patient(hadm_id)
            self.train_one_batch[i, :] = one_data
            one_data_logit = self.hetro_model.assign_multi_hot(hadm_id)
            self.logit_one_batch[i, :] = one_data_logit

    def train(self):
        """
        train the system
        """
        iteration = np.int(np.floor(np.float(self.length_train_hadm)/self.batch_size))
        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                self.get_batch_train(i+self.batch_size)
                self.err_ = self.sess.run([self.cross_entropy, self.train_step_cross_entropy],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.input_y_diag: self.logit_one_batch})
                print(self.err_[0])

    def test(self):
        """
        return test f1 score
        """
        length_test_hadmid = len(self.data_process.test_hadm_id)
        self.test_output = np.zeros((length_test_hadmid, self.item_size))
        self.test_logit_data = np.zeros((length_test_hadmid, self.diagnosis_size))
        index = 0
        for i in self.data_process.test_hadm_id:
            one_data = self.hetro_model.assign_value_patient(i)
            self.test_output[index, :] = one_data
            test_one_data_logit = self.hetro_model.assign_multi_hot(i)
            self.test_logit_data[index, :] = test_one_data_logit
            index += 1
        self.logit_output = self.sess.run(self.logit_softmax, feed_dict={self.input_x: self.test_output})
        self.tp_rate_total = []
        self.fp_rate_total = []
        self.tp_rate_roc = []
        self.fp_rate_roc = []
        self.threshold = 0.0
        iter = 0
        while(self.threshold<1.005):
            print(iter)
            for k in range(length_test_hadmid):
                test_sample = self.logit_output[k, :]
                actual_logit = self.test_logit_data[k,:]
                detect = np.where(test_sample > self.threshold)
                actual = np.where(actual_logit > 0.1)
                actual_neg = np.where(actual_logit < 0.1)
                correct_detect = len([i for i in detect[0] if i in actual[0]])
                uncorrect_detect = len([i for i in detect[0] if i in actual_neg[0]])
                tp_rate = float(correct_detect) / len(actual[0])
                fp_rate = float(uncorrect_detect) / len(actual_neg[0])
                self.tp_rate_total.append(tp_rate)
                self.fp_rate_total.append(fp_rate)
            self.tp_test = np.mean(self.tp_rate_total)
            self.fp_test = np.mean(self.fp_rate_total)
            self.tp_rate_roc.append(self.tp_test)
            self.fp_rate_roc.append(self.fp_test)
            self.tp_rate_total = []
            self.fp_rate_total = []
            self.threshold += self.resolution
            iter += 1

    def diag_accur(self):
        self.frequnce = np.zeros(len(self.kg.dic_diag.keys()))
        self.diag_f1_score = np.zeros(len(self.kg.dic_diag.keys()))
        for i in range(len(self.kg.dic_diag.keys())):
            num_diag = 0
            true_positives = 0
            true_negatives = 0
            false_negative = 0
            false_positive = 0
            for j in range(len(self.data_process.test_hadm_id)):
                if self.test_logit_data[j,i] >0.1:
                    num_diag += 1
                if self.logit_output[j,i] > self.threshold_diag and self.test_logit_data[j,i] > 0.1:
                    true_positives += 1
                if self.logit_output[j,i] < self.threshold_diag and self.test_logit_data[j,i] < 0.1:
                    true_negatives += 1
                if self.logit_output[j,i] > self.threshold_diag and self.test_logit_data[j,i] < 0.1:
                    false_positive += 1
                if self.logit_output[j, i] < self.threshold_diag and self.test_logit_data[j, i] > 0.1:
                    false_negative += 1

            if true_positives == 0:
                self.diag_f1_score[i] = np.float(0)
            else:
                precision = np.float(true_positives)/(true_positives+false_positive)
                recall = np.float(true_positives)/(false_negative+true_positives)
                self.diag_f1_score[i] = 2*(precision*recall)/(precision+recall)
            self.frequnce[i] = num_diag

    def write_file(self,file_name_tp,file_name_fp):
        with open(file_name_tp,"w") as output:
            output.write(str(self.tp_rate_roc))
        with open(file_name_fp,"w") as output:
            output.write(str(self.fp_rate_roc))

    def cal_auc(self):
        area = 0
        self.tp_rate_roc.sort()
        self.fp_rate_roc.sort()
        for i in range(len(self.tp_rate_roc) - 1):
            x = self.fp_rate_roc[i + 1] - self.fp_rate_roc[i]
            y = (self.tp_rate_roc[i + 1] + self.tp_rate_roc[i]) / 2
            area += x * y
        return area