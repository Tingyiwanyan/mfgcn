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
        self.cross_validation_folder = np.int(10)
        self.kg = kg
        #self.data_patient_num = len(kg.dic_patient_addmission.keys())
        self.data_hadm_id = kg.dic_patient.keys()
        #self.train_num = np.int(np.floor(self.data_patient_num*self.train_percent))
        self.train_hadm_id = []
        self.train_patient = []
        self.test_patient = []
        self.test_hadm_id = []
        self.data_patient = []
        self.data_hadm_id = []


    def cross_validation_seperate(self,index):
        """
        split first patient encounter into 10 fold
        part: which part to use as validation
        """
        self.part_test_num = np.int(self.cross_validation_folder*self.test_percent)
        self.part_train_num = np.int(self.cross_validation_folder*self.train_percent)
        self.split_test = []
        self.split_train = []
        self.split = np.array_split(self.kg.dic_patient.keys(),self.cross_validation_folder)
        self.split_test_data = []
        self.split_train_data = []
        for i in range(self.part_test_num):
            self.split_test.append(index+i)
        for i in range(self.part_test_num):
            if self.split_test[i] == self.cross_validation_folder or self.split_test[i] > self.cross_validation_folder:
                self.split_test[i] = self.split_test[i] - self.cross_validation_folder
        self.split_train = [i for i in range(self.cross_validation_folder) if i not in self.split_test]
        for i in self.split_test:
            self.split_test_data = self.split_test_data + list(self.split[i])
        for i in self.split_train:
            self.split_train_data = self.split_train_data + list(self.split[i])

        self.train_hadm_id = self.split_train_data
        self.test_hadm_id = self.split_test_data

    def seperate_train_test(self):
        """
        prepare train and test data set
        """
        for i in self.kg.dic_patient_addmission.keys():
            time_len = len(self.kg.dic_patient_addmission[i]['time_series'])
            if time_len > 1 or time_len == 1:
                self.data_patient.append(i)
                #for j in self.kg.dic_patient_addmission[i]['time_series']:
                 #   self.train_hadm_id.append(j)
        self.data_patient_num = len(self.data_patient)
        self.train_num = np.int(np.floor(self.data_patient_num * self.train_percent))
        self.train_patient = self.data_patient[0:self.train_num]
        for i in self.train_patient:
            for j in self.kg.dic_patient_addmission[i]['time_series']:
               self.train_hadm_id.append(j)

        self.test_patient = [i for i in self.data_patient if i not in self.train_patient]
        self.test_hadm_id = [i for i in self.kg.dic_patient.keys() if i not in self.train_hadm_id]

        #self.test = [i for i in self.data_hadm_id if i not in self.train_hadm_id]




