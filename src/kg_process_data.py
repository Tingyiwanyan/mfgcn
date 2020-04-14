import os
import json
import numpy as np
import random

class kg_process_data():
    """
    process heterogeneous kg graph data, divide into train and test data set
    """
    def __init__(self,kg):
        self.train_percent = 0.7
        self.test_percent = 0.3
        self.batch_size = 16
        self.kg = kg
        self.data_patient_num = len(kg.dic_patient_addmission.keys())
        self.data_hadm_id = kg.dic_patient.keys()
        self.train_num = np.int(np.floor(self.data_patient_num*self.train_percent))
        self.train_hadm_id = []
        self.train_patient = []
        self.test_patient = []
        self.test_hadm_id = []



    def seperate_train_test(self):
        """
        prepare train and test data set
        """
        for i in self.kg.dic_patient_addmission.keys():
            time_len = len(self.kg.dic_patient_addmission[i]['time_series'])
            if time_len > 3 or time_len == 3:
                self.train_patient.append(i)
                for j in self.kg.dic_patient_addmission[i]['time_series']:
                    self.train_hadm_id.append(j)
            cur_length = len(self.train_patient)
            if cur_length > self.train_num:
                break


        self.test = [i for i in self.data_hadm_id if i not in self.train_hadm_id]



