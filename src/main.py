from model import model
from utils import utils
from data_load import Data_loading
from evaluation import evaluation

if __name__ == "__main__":
  """
  utils input arguments: (option for data set, option for using different model, option for doing different tasks)
  """
  utils = utils(2,3,2)
  utils.config_train_test()
  utils.config_model()
  utils.init_walk_prob()
  utils.train()
  #evl = evaluation(utils)
  #evl.evaluate(utils)