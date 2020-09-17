from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import pickle
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import Prediction_model.classification_model as models
#from deep_fairness.fairyted import Fairytale
from Prediction_model.helper import load_pickle, sample_indices, cvt, make_minibatch, counterfactual_loss, calc_acc, MacOSFile, variation_loss, pad

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim




class Experiment_diversifier(object):

  def __init__(self, config):
    super(Experiment_diversifier, self).__init__()
    self.config = config

    # assert(self.config['model_params']['in_size'] == self.config['trans_dim'] + 
    #   self.config['u_dim'] + self.config['view_dim'] + self.config['a_dim'])

    # Use appropritate device
    if self.config['gpu_num'] < 0:
      self.device = torch.device('cpu')
    else:
      self.device = torch.device('gpu:{0}'.format(self.config['gpu_num']))

    
    text_data_dict_path = os.path.join(self.config['input_path'], self.config['text_input_filename'])
    text_data_dict = load_pickle(text_data_dict_path)
    for key in text_data_dict:
      text_data_dict[key] = np.array(text_data_dict[key])
    self.text_data = text_data_dict



    vid_data_dict_path = os.path.join(self.config['input_path'], self.config['vid_input_filename'])
    vid_data_dict = load_pickle(vid_data_dict_path)
    for key in vid_data_dict:
      vid_data_dict[key] = np.array(vid_data_dict[key])
    self.vid_data = vid_data_dict
    # if self.config['use_simulated_data']:
    #   self.fairmodel = Fairytale()
    # else:
    #   self.fairmodel = Fairytale(data=data_dict)

    # Prepare the neural network and others
    self.total_epoch = 0
    self.best_loss = float('inf')
    self.model = getattr(models, self.config['model_type'])(**self.config['model_params']).to(device=self.device)
    self.loss_fn = getattr(nn, self.config['loss_function_name'])()
    self.optimizer = getattr(optim, self.config['optimizer'])(self.model.parameters(),**self.config['optimizer_params'])
    self.scheduler = getattr(optim.lr_scheduler, self.config['scheduler'])(self.optimizer, **self.config['scheduler_params'])
    self.relu = nn.ReLU()
    #self.cf_loss = counterfactual_loss
    self.text_divloss = variation_loss
    self.vid_divloss = variation_loss

    self.rating_names = ['beautiful', 'confusing', 'courageous', 'fascinating', 'funny', 'informative', 'ingenious', 'inspiring', 'jaw-dropping', 'longwinded', 'obnoxious', 'ok', 'persuasive', 'unconvincing']
    self.sub_ratings = ['fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']
    self.rating_index = [self.rating_names.index(x) for x in self.sub_ratings]

  def set_random_seed(self):      
    seed = self.config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




  def test_model(self, orig_data, test_idx):
    self.model.eval()

    
    if self.config["input_type"] == "transcript_only":
      inputs = orig_data['input'][test_idx,:207]
    elif self.config["input_type"] == "transcript_plus_video":
      col_list = np.concatenate((range(207),range(208,408)))
      inputs = orig_data['input'][test_idx,:][:,col_list]
    elif self.config["input_type"] == "video_only":
      col_list = np.concatenate((range(200,207),range(208,408)))
      inputs = orig_data['input'][test_idx,:][:,col_list]
    labels = orig_data['label'][test_idx,:]
    

    #print(inputs)
    with torch.set_grad_enabled(False):
      outputs = self.model(inputs)
      x,total_acc = calc_acc(outputs, labels)
    #print(outputs)
    average_test_acc = np.mean(total_acc.numpy(),axis=0)
    print("Test Accuracy is :",average_test_acc)
    print("Model Accuracy: ",np.mean(average_test_acc))
    return x,average_test_acc,outputs


  def train_model(self, orig_data, train_idx, dev_idx, 
    max_epochs=10, max_iter=10, use_textdiv=False, use_viddiv=False,minibatch_size=10):


    
    since = time.time()

    best_model_wts = copy.deepcopy(self.model.state_dict())
    text_divloss_list = []
    vid_divloss_list = []
    trainloss_list = []
    for epoch in range(max_epochs):
      
      print('Epoch {}/{}'.format(epoch, max_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
        if phase == 'train':
          self.model.train()  # Set model to training mode
          indices = train_idx
        else:
          self.model.eval()   # Set model to evaluate mode
          indices = dev_idx

        running_loss = 0.0
        text_divloss = 0
        vid_divloss = 0
        # Iterate over data.
        for iter, a_batch in enumerate(make_minibatch(indices, minibatch_size)):
          
          if iter > max_iter:
            break

          if self.config["input_type"] == "transcript_only":
            inputs = orig_data['input'][a_batch,:207]
          elif self.config["input_type"] == "transcript_plus_video":
            col_list = np.concatenate((range(207),range(208,408)))
            inputs = orig_data['input'][a_batch,:][:,col_list]
          elif self.config["input_type"] == "video_only":
            col_list = np.concatenate((range(200,207),range(208,408)))
            inputs = orig_data['input'][a_batch,:][:,col_list]

          if self.config['use_binned_text_diversity']:
            text_diversity = orig_data['binned_text_diversity'][a_batch]
          else:
            text_diversity = orig_data['text_diversity'][a_batch]
          vid_diversity = orig_data['vid_diversity'][a_batch]
          labels = orig_data['label'][a_batch,:]


          # zero the parameter gradients
          self.optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            # outputs.register_hook(lambda grad: print("loss grad",grad))

            #loss.register_hook(lambda: grad)
            

            # backward + optimize only if in training phase
            if phase == 'train':
                if use_textdiv:
                  #print("hooo=====")
                  # print()
                  #print(text_diversity)
                  
                  divloss_list = self.text_divloss(outputs,labels,text_diversity,self.config['epsilon'])
                  temp_tdloss = torch.mean(self.relu(divloss_list))
                  # temp_tdloss.register_hook(lambda grad: print("div loss grad ",grad))
                  #print(loss,'before')
                  loss = loss+ self.config['lambda']*temp_tdloss
                  #print(loss,'after')
                  text_divloss = text_divloss + self.config['lambda']*temp_tdloss

                if use_viddiv:
                  #print("hooo=====")
                  # print()
                  #print(text_diversity)
                  
                  divloss_list = self.vid_divloss(outputs,labels,vid_diversity,self.config['epsilon'])
                  temp_tdloss = torch.mean(self.relu(divloss_list))
                  # temp_tdloss.register_hook(lambda grad: print("div loss grad ",grad))
                  #print(loss,'before')
                  loss = loss+ self.config['lambda']*temp_tdloss
                  #print(loss,'after')
                  vid_divloss = vid_divloss + self.config['lambda']*temp_tdloss
                loss.backward()
                self.optimizer.step()                
                
          # statistics
          running_loss += loss.item()
        if phase=='train':
          text_divloss_list.append(text_divloss)
          vid_divloss_list.append(vid_divloss)
          trainloss_list.append(running_loss)

          

        epoch_loss = running_loss / (max_iter)
        epoch_text_divloss = text_divloss/ (max_iter)#* minibatch_size**2)
        epoch_vid_divloss = vid_divloss/ (max_iter)
        print('{} Loss: {:.4f} '.format(phase, epoch_loss))
        if use_textdiv and phase=='train':
          print('{} Loss: {:.4f} '.format('textdiv', epoch_text_divloss))
        if use_viddiv and phase=='train':
          print('{} Loss: {:.4f} '.format('viddiv', epoch_vid_divloss))

        # deep copy the model
        if phase == 'val' and epoch_loss < self.best_loss:
          self.best_loss = epoch_loss
          best_model_wts = copy.deepcopy(self.model.state_dict())

      

    time_elapsed = time.time() - since

    self.total_epoch += epoch 
    print('Training complete in {:.0f}m {:.0f}s (total epoch = {})'.format(
      time_elapsed // 60, time_elapsed % 60, self.total_epoch))

    #plot divloss
    #plt.subplot(1,3,1)
    if use_textdiv:
      plt.plot(range(len(text_divloss_list)), text_divloss_list,label='Verbal HEM loss')
    
    if use_viddiv:
      plt.plot(range(len(vid_divloss_list)), vid_divloss_list,label='Non-verbal HEM loss')
    
    plt.plot(range(len(trainloss_list)), trainloss_list,label="Total Train Loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    plt.tight_layout()
    plt.savefig('./Plots/Loss_{}_{}_{}_{}_{}.pdf'.format(self.config["trainer_params"]["use_textdiv"],self.config["trainer_params"]["use_viddiv"],self.config['num_bin'],
      self.config['epsilon'], self.config['lambda']))
    #plt.show()
    plt.close()
    
    # save last element of text_divloss list
    with open("./Output/text_divloss_last.txt", "w") as text_file:
      text_file.write("{:.4f} {} {} {} {} {}".format(text_divloss_list[-1], self.config["trainer_params"]["use_textdiv"],self.config["trainer_params"]["use_viddiv"],
        self.config['num_bin'],  self.config['epsilon'], self.config['lambda']))

    # load best model weights
    self.model.load_state_dict(best_model_wts)
    with open('Output/loss_fn.pkl','wb') as f:
      pickle.dump(text_divloss_list,f)

  def save_model(self, model_filepath):
    checkpoint = {
      'total_epoch':self.total_epoch,
      'model_state_dict':self.model.state_dict(),
      'optimizer_state_dict':self.optimizer.state_dict(),
      'scheduler_state_dict':self.scheduler.state_dict(),
      'best_loss':self.best_loss
    }
    torch.save(checkpoint, model_filepath)
    

  def load_model(self, model_filepath):
    checkpoint = torch.load(model_filepath)
    self.total_epoch = checkpoint['total_epoch']
    self.best_loss = checkpoint['best_loss']
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  def split(self,num_inp):
    
    # Divide into train/dev/test
    # ==========================
    print("splitting in process for ",num_inp," data.")
    train_idx, dev_idx, test_idx = sample_indices(num_inp)
    index_filename = os.path.join(self.config['output_path'], self.config['split_filename'])
    pickle.dump((train_idx, dev_idx, test_idx), open(index_filename, 'wb'))
    return train_idx, dev_idx, test_idx

  def load_split_index(self):
    index_filename = os.path.join(self.config['output_path'], self.config['split_filename'])
    train_idx, dev_idx, test_idx = pickle.load(open(index_filename, 'rb'))
    return train_idx, dev_idx, test_idx


  def run(self):
    '''
    Runs the experiment
    '''
    if self.config["create_jointdata"]:
      print("Joining Data")
      new_data_dic = pad(self.vid_data,self.text_data,self.config['num_eigen_tokeep'],self.config['max_vid_len'])
      pickle.dump(new_data_dic,open(self.config['input_path']+'/'+self.config["joint_filename"],'wb'))
      print(new_data_dic.keys())
    else:
      new_data_dic = load_pickle(self.config['input_path']+'/'+self.config["joint_filename"])

    #Process data for input of the model
    # ==========================
    orig_data = {}
    
    orig_data["input"] = new_data_dic['input']
    orig_data["label"] = new_data_dic['rating'][:,self.rating_index]
    orig_data["text_diversity"] = new_data_dic['text_diversity']
    orig_data["vid_diversity"] = new_data_dic['vid_diversity'] 
     
    # Divide into train/dev/test
    # ==========================
    if self.config["need_split"]:
      train_idx, dev_idx, test_idx = self.split(orig_data['input'].shape[0])
      
    else:
      train_idx, dev_idx, test_idx = self.load_split_index()

    # Binned diversity calculation
    # ==========================
    self.min_train_div = min(new_data_dic['text_diversity'][train_idx])
    self.max_train_div = max(new_data_dic['text_diversity'][train_idx])
    diff = self.max_train_div - self.min_train_div
    bin_len = diff/self.config['num_bin']
    binned_text_diversity = np.array([int(x/bin_len)*bin_len+bin_len/2.0 for x in orig_data['text_diversity']])
    orig_data['binned_text_diversity'] = binned_text_diversity
    print("Splitting done")
    

    # Send to appropriate device
    # ==========================
    for a_key in orig_data:
      orig_data[a_key] = torch.from_numpy(orig_data[a_key].astype(np.float32)).to(device=self.device)
    print("Data Load done") 



    # Neural network training part
    # ============================
    if self.config['train_neural_network']:
      #print("Training Model number: ",model_id)
      nn_filename = os.path.join(self.config['output_path'], self.config['neural_network_model_filename'])
      if self.config['load_nn_from_file']:
        self.load_model(nn_filename)
      print(orig_data['input'].shape)
      self.train_model(orig_data, train_idx, dev_idx, **self.config['trainer_params'])
      self.save_model(nn_filename)

    # Test Neural Network
    # ===================
    if self.config['test_neural_network']:
      nn_filename = os.path.join(self.config['output_path'], self.config['load_nn_filename'])
      self.load_model(nn_filename)
      op,average_test_acc, un_norm_op = self.test_model(orig_data, test_idx)
      
      #save test accuracy
      with open('Output/Acc/acc_'+self.config["input_type"]+'_{}_{}_{}_{}_{}'.format(self.config["trainer_params"]["use_textdiv"],self.config["trainer_params"]["use_viddiv"],
  self.config['num_bin'],  self.config['epsilon'], self.config['lambda'])+'.pkl','wb') as g:
        pickle.dump((average_test_acc,np.mean(average_test_acc)) ,g)



      op = op.numpy()
      inp =  new_data_dic['input'][test_idx,:]
      data_dict_predict, data_dict_true = {},{}
      data_dict_true['transcript'] = inp[:,:200]
      data_dict_predict['transcript'] = inp[:,:200]

      data_dict_true['a'] = inp[:,200:207]
      data_dict_true['view'] = inp[:,207]
      
      data_dict_predict['a'] = inp[:,200:207]
      data_dict_predict['view'] = inp[:,207]
      
      data_dict_true['rating'] = orig_data['label'][test_idx,:].numpy()
      data_dict_predict['rating'] = op

      m = nn.Sigmoid()

      data_dict_predict['un_norm_op'] = m(un_norm_op)
      self.data_dict_predict = data_dict_predict
      self.data_dict_true = data_dict_true 
      with open('Output/test_output_'+self.config["input_type"]+'_{}_{}_{}_{}_{}'.format(self.config["trainer_params"]["use_textdiv"],self.config["trainer_params"]["use_viddiv"],
  self.config['num_bin'],  self.config['epsilon'], self.config['lambda'])+'.pkl','wb') as f:
        pickle.dump((data_dict_predict,data_dict_true),f) 


    


