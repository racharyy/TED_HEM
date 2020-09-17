import argparse
import yaml
import pickle
import os
#import pymc3 as pm 
import matplotlib.pyplot as plt
import numpy as np 
#import theano
#import theano.tensor as tt
from scipy import stats
#import pymc3 as pm 
import sys
sys.path.insert(0,'../')
#from deep_fairness.simul_data import model1
#from deep_fairness.fairyted import Fairytale
#from deep_fairness.counterfactual_generate import counterfactual_sample
#from deep_fairness.pymc_model_multivariate import model_fit
from Prediction_model.helper import load_pickle, dict_to_concat_data
from Prediction_model.experiment_diversifier import Experiment_diversifier

#from dummyfordiscussion import *

def data_load():
  data_dict = load_pickle('../Data/converted_data_dict.pkl')
  for key in data_dict:
    print(len(data_dict[key]),key)
    data_dict[key] = np.array(data_dict[key])
  fairmodel = Fairytale()
  mf,trace=fairmodel.fit_params()
  data_with_u, cfsample = fairmodel.counterfactual_generate(trace)
  orig_concat_data, cf_concat_data = fairmodel.create_concat_data(data_with_u, cfsample)


def main():
  parser = argparse.ArgumentParser('Train and Evaluate Neural Networks')
  parser.add_argument('--conf', dest='config_filepath', 
    help='Full path to the configuration file')

  parser.add_argument('--bin', dest='num_bin', type=int, default=None,
    help='number of bins for diversity')
  parser.add_argument('--eps', dest='eps',  type=float, default=None,
    help='epsilon')
  parser.add_argument('--lam', dest='lam',  type=float, default=None,
    help='lambda for diversity loss')
  parser.add_argument('--div', dest='div',  type=bool, default=False,
    help='Whether to use HEM during training')
  parser.add_argument('--split', dest='split',  type=bool, default=None,
    help='Whether to split the data before training')

  args = parser.parse_args()
  
  if args.config_filepath and os.path.exists(args.config_filepath):
    conf = yaml.load(open(args.config_filepath), Loader=yaml.FullLoader)
  else:
    raise Exception('Config file not found')

  if args.num_bin : conf['numb_bin'] = args.num_bin
  if args.eps : conf['epsilon'] = args.eps
  if args.lam : conf['lambda'] = args.lam
  if args.split : conf[ "need_split"] = args.split
  if args.div :
    print("HEM loss added")
    conf["trainer_params"]["use_textdiv"] =  args.div
    conf["trainer_params"]["use_viddiv"] =  args.div
 
    
  exp = Experiment_diversifier(conf)
  exp.run()
  # if conf["test_neural_network"]:
  #   data_dict_predict_ar.append(exp.data_dict_predict['rating'])

  # if conf["test_neural_network"]:
  #   data_dict_predict_rating = 1*(np.mean(np.array(data_dict_predict_ar),axis=0)>=0.5)
  #   exp.data_dict_predict['rating'] = data_dict_predict_rating
  #   acc=np.mean(data_dict_predict_rating==exp.data_dict_true['rating'],axis=0)
  #   print("Accuracy is: \n")
  #   print(acc)
  #   print("Average Accuracy: ",np.mean(acc))
  #   with open('Output/test_output_combined.pkl','wb') as f:
  #       pickle.dump((exp.data_dict_predict,exp.data_dict_true),f)

if __name__=='__main__':
  main()
