import tensorflow as tf
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import networkx as nx

class visualization(object):
    def __init__(self,utils,evaluation):