from model import model
from utils import utils
from data_load import Data_loading
from evaluation import evaluation
from visualization import visualization
import numpy as np


if __name__ == "__main__":
  """
  utils input arguments: (option for data set, option for using different model, option for doing different tasks,
  option for choosing random walk strategy,option of whether to add structure)
  """
  utils = utils(3,3,2,2,1)

  utils.config_train_test()
  utils.config_model()
  utils.init_walk_prob()
  utils.train()
  evl = evaluation(utils,2)
  #evl.evaluate(utils)
  #vis = visualization(utils,evl)
  #vis.get_2d_rep()
  #vis.plot_2d()