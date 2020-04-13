import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
import pandas as pd
from kg_model import hetero_model
from LSTM_model import LSTM_model
from kg_process_data import kg_process_data

class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self):
        file_path = '/home/tingyi/MIMIC'
        self.diagnosis = file_path + '/DIAGNOSES_ICD.csv'
        self.diagnosis_d = file_path + '/D_ICD_DIAGNOSES.csv'
        self.prescription = file_path + '/PRESCRIPTIONS.csv'
        self.charteve = file_path + '/CHARTEVENTS.csv'
        self.d_item = file_path + '/D_ITEMS.csv'
        self.noteevents = file_path + '/NOTEEVENTS.csv'
        self.proc_icd = file_path + '/PROCEDURES_ICD.csv'
        self.read_diagnosis()
        self.read_charteve()
        self.read_diagnosis_d()
        self.read_prescription()
        self.read_ditem()
        #self.read_proc_icd()

    def read_diagnosis(self):
        self.diag = pd.read_csv(self.diagnosis)
        self.diag_ar = np.array(self.diag)

    def read_diagnosis_d(self):
        self.diag_d = pd.read_csv(self.diagnosis_d)
        self.diag_d_ar = np.array(self.diag_d)

    def read_prescription(self):
        self.pres = pd.read_csv(self.prescription)

    def read_charteve(self):
        self.char = pd.read_csv(self.charteve,chunksize=3000000)
        self.char_ar = np.array(self.char.get_chunk())
        self.num_char = self.char_ar.shape[0]

    def read_ditem(self):
        self.d_item = pd.read_csv(self.d_item)
        self.d_item_ar = np.array(self.d_item)

    def read_noteevent(self):
        self.note = pd.read_csv(self.noteevents,chunksize=1000)

    def read_proc_icd(self):
        self.proc_icd = pd.read_csv(self.proc_icd)

    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_diag = {}
        self.dic_item = {}
        self.dic_patient_addmission = {}
        index_item = 0
        index_diag = 0
        for i in range(self.num_char):
            itemid = self.char_ar[i][4]
            value = self.char_ar[i][8]
            hadm_id = self.char_ar[i][2]
            patient_id = self.char_ar[i][1]
            date_time = self.char_ar[i][5].split(' ')
            date = [np.int(i) for i in date_time[0].split('-')]
            date_value = date[0]*10000+date[1]*100+date[2]
            time = [np.int(i) for i in date_time[1].split(':')]
            time_value = time[0]*100 + time[1]
            date_time_value = date_value*10000+time_value
            if itemid not in self.dic_item:
                self.dic_item[itemid] = {}
                self.dic_item[itemid]['nodetype'] = 'item'
                self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)
                self.dic_item[itemid]['item_index'] = index_item
                self.dic_item[itemid].setdefault('value', []).append(value)
                #self.dic_item[itemid]['mean_value'] = []
                index_item += 1
            else:
                self.dic_item[itemid].setdefault('value', []).append(value)
                if hadm_id not in self.dic_item[itemid]['neighbor_patient']:
                    self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)

            if hadm_id not in self.dic_patient:
                self.dic_patient[hadm_id] = {}
                self.dic_patient[hadm_id]['itemid'] = {}
                self.dic_patient[hadm_id]['nodetype'] = 'patient'
                self.dic_patient[hadm_id]['next_admission'] = None
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid, []).append(value)
                #self.dic_patient[patient_id]['neighbor_presc'] = {}
            else:
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid,[]).append(value)

            if patient_id not in self.dic_patient_addmission:
                self.dic_patient_addmission[patient_id] = {}
                self.dic_patient_addmission[patient_id][hadm_id] = {}
                self.dic_patient_addmission[patient_id][hadm_id]['date_time'] = date_time
                self.dic_patient_addmission[patient_id][hadm_id]['date'] = date
                self.dic_patient_addmission[patient_id][hadm_id]['date_value'] = date_value
                self.dic_patient_addmission[patient_id][hadm_id]['time'] = time
                self.dic_patient_addmission[patient_id][hadm_id]['time_value'] = time_value
                self.dic_patient_addmission[patient_id][hadm_id]['date_time_value'] = date_time_value
                self.dic_patient_addmission[patient_id].setdefault('time_series',[]).append(hadm_id)
            else:
                if hadm_id not in self.dic_patient_addmission[patient_id]['time_series']:
                    self.dic_patient_addmission[patient_id][hadm_id] = {}
                    self.dic_patient_addmission[patient_id][hadm_id]['date_time'] = date_time
                    self.dic_patient_addmission[patient_id][hadm_id]['date'] = date
                    self.dic_patient_addmission[patient_id][hadm_id]['time'] = time
                    self.dic_patient_addmission[patient_id][hadm_id]['date_value'] = date_value
                    self.dic_patient_addmission[patient_id][hadm_id]['time_value'] = time_value
                    self.dic_patient_addmission[patient_id][hadm_id]['date_time_value'] = date_time_value
                    index = 0
                    flag = 0
                    for i in self.dic_patient_addmission[patient_id]['time_series']:
                        if date_time_value < self.dic_patient_addmission[patient_id][i]['date_time_value']:
                            self.dic_patient_addmission[patient_id]['time_series'].insert(index,hadm_id)
                            self.dic_patient[hadm_id]['next_admission'] = i
                            if not index == 0:
                                self.dic_patient[self.dic_patient_addmission[patient_id]['time_series'][index-1]]['next_admission'] = hadm_id
                            flag = 1
                            break
                        index += 1
                    if flag == 0:
                        self.dic_patient[self.dic_patient_addmission[patient_id]['time_series'][-1]]['next_admission'] = hadm_id
                        self.dic_patient_addmission[patient_id].setdefault('time_series', []).append(hadm_id)



            #self.dic_patient.setdefault('itemid',[]).append([])

        for i in range(self.diag_ar.shape[0]):
            hadm_id = self.diag_ar[i][2]
            diag_icd = self.diag_ar[i][4]
            if hadm_id in self.dic_patient:
                if diag_icd not in self.dic_diag:
                    self.dic_diag[diag_icd] = {}
                    self.dic_diag[diag_icd].setdefault('neighbor_patient', []).append(hadm_id)
                    self.dic_diag[diag_icd]['nodetype'] = 'diagnosis'
                    self.dic_diag[diag_icd]['diag_index'] = index_diag
                    self.dic_patient[hadm_id].setdefault('neighbor_diag', []).append(diag_icd)
                    index_diag += 1
                else:
                    self.dic_patient[hadm_id].setdefault('neighbor_diag',[]).append(diag_icd)
                    self.dic_diag[diag_icd].setdefault('neighbor_patient',[]).append(hadm_id)

        for i in self.dic_item.keys():
            self.dic_item[i]['mean_value'] = np.mean(self.dic_item[i]['value'])
            self.dic_item[i]['std'] = np.std(self.dic_item[i]['value'])




    def create_kg(self):
        self.g = nx.DiGraph()
        for i in range(self.num_char):
            patient_id = self.char_ar[i][1]/home/tingyi/ecgtoolkit-cs-git/ECGToolkit/libs/ECGConversion/MUSEXML/MUSEXML/MUSEXMLFormat.cs
            itemid = self.char_ar[i][4]
            value = self.char_ar[i][8]
            itemid_list = np.where(self.d_item_ar == itemid)
            diag_list = np.where(self.diag_ar[:,1] == patient_id)
            diag_icd9_list = self.diag_ar[:,4][diag_list]
            diag_d_list = [np.where(self.diag_d_ar[:,1] == diag_icd9_list[x])[0] for x in range(diag_icd9_list.shape[0])]
            """
            Add patient node
            """
            self.g.add_node(patient_id, item_id=itemid)
            self.g.add_node(patient_id, test_value=value)
            self.g.add_node(patient_id, node_type='patient')
            self.g.add_node(patient_id, itemid_list=itemid_list)
            self.g.add_node(itemid, node_type='ICD9')
            """
            Add diagnosis ICD9 node
            """
            self.g.add_edge(patient_id, itemid, type='')




if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.create_kg_dic()
    hetro_model = hetero_model(kg)
    process_data = kg_process_data(kg)
    process_data.seperate_train_test()
    LSTM_model = LSTM_model(kg,hetro_model,process_data)

