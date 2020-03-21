import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
import pandas as pd
from kg_model import hetero_model

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
        self.char = pd.read_csv(self.charteve,chunksize=300000)
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
        for i in range(self.num_char):
            itemid = self.char_ar[i][4]
            value = self.char_ar[i][8]
            hadm_id = self.char_ar[i][2]
            if hadm_id not in self.dic_patient:
                self.dic_patient[hadm_id] = {}
                self.dic_patient[hadm_id]['itemid'] = {}
                self.dic_patient[hadm_id]['nodetype'] = 'patient'
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid, []).append(value)
                #self.dic_patient[patient_id]['neighbor_presc'] = {}
            else:
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid,[]).append(value)

            if itemid not in self.dic_item:
                self.dic_item[itemid] = {}
                self.dic_item[itemid]['nodetype'] = 'item'
                self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)
            else:
                if hadm_id not in self.dic_item[itemid]['neighbor_patient']:
                    self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)
            #self.dic_patient.setdefault('itemid',[]).append([])

        for i in range(self.diag_ar.shape[0]):
            hadm_id = self.diag_ar[i][2]
            diag_icd = self.diag_ar[i][4]
            if hadm_id in self.dic_patient:
                if diag_icd not in self.dic_diag:
                    self.dic_diag[diag_icd] = {}
                    self.dic_diag[diag_icd].setdefault('neighbor_patient', []).append(hadm_id)
                    self.dic_diag[diag_icd]['nodetype'] = 'diagnosis'
                else:
                    self.dic_patient[hadm_id].setdefault('neighbor_diag',[]).append(diag_icd)
                    self.dic_diag[diag_icd].setdefault('neighbor_patient',[]).append(hadm_id)





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
    model = hetero_model(kg)

