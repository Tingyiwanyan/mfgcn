import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby

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
        self.batch_size = 16
        self.time_sequence = 3
        self.latent_dim = 100
        self.latent_dim_cell_state = 100
        self.epoch = 6
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))