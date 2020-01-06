from model import model
from utils import utils
from data_load import Data_loading
from evaluation import evaluation

if __name__ == "__main__":

  utils = utils(1,1)
  utils.config_train_test()
  utils.config_model()
  #utils.train()
  #evl = evaluation(utils)
  #evl.evaluate(utils)