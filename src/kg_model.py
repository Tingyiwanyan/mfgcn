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
        self.walk_length = 15
        self.latent_dim = 100
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size =len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.walk = []

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


    def extract_meta_path(self, node_type, start_index):
        """
        Perform metapath from different starting node type
        :param node_type: node_type
        :param start_index: ID for starting node
        :return: meta path
        """
        if node_type == "patient":

        if node_type == "item":


