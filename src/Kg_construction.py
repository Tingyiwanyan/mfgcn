import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
import pandas as pd

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
        self.read_proc_icd()

    def read_diagnosis(self):
        self.diag = pd.read_csv(self.diagnosis)

    def read_diagnosis_d(self):
        self.diag_d = pd.read_csv(self.diagnosis_d)

    def read_prescription(self):
        self.pres = pd.read_csv(self.prescription)

    def read_charteve(self):
        self.char = pd.read_csv(self.charteve,chunksize=10000)

    def read_ditem(self):
        self.d_item = pd.read_csv(self.d_item)

    def read_noteevent(self):
        self.note = pd.read_csv(self.noteevents,chunksize=10000)

    def read_proc_icd(self):
        self.proc_icd = pd.read_csv(self.proc_icd)


    def create_kg(self):
        self.g = nx.DiGraph()
        self.g.add_node()


if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.read_charteve()
    kg.read_diagnosis()
    kg.read_proc_icd()

