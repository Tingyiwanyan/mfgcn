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
    def __init__(self):
        self.time_step