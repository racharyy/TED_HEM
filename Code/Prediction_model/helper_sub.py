import torch
import random
import copy
import random
import math
import os
import pickle
import numpy as np
import time
import glob
import torch.nn as nn
import itertools as it
import pandas as pd
import pickle




def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data






def convert_dict_to_categorical_sub(df):

    # convert dataframe from one hot encoded dict to categorical for gender rance
    df_for_metric = pd.DataFrame({})
    df_for_metric['view'] = df['view']
    
    protected_df = pd.DataFrame(data = df['a'], index=None, 
                                columns = ['female', 'male', 'gender_other', 'white', 'black_or_african_american',
                                                               'asian','race_other'] )
    
    gender_df = protected_df[['male','female','gender_other']]
    race_df = protected_df[['white', 'black_or_african_american','asian','race_other']]
    

    gender_race = pd.DataFrame({})
    gender_race['gender'] = gender_df.idxmax(1).tolist()
    gender_race['race'] =  race_df.idxmax(1).tolist()
    
    df_for_metric['gender'] = gender_race['gender']
    df_for_metric['race'] = gender_race['race']

    
    rating_df = pd.DataFrame(data = df['rating'], index=None, columns = ['fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok'] )
    
    df_for_metric[['fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']] = rating_df[['fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]


    # for transcripts use 50 columns for doc2vec50
    array = df['transcript']
    temp_transcript = pd.DataFrame(data=array[:,:],    # values
                                   index=None,    # 1st column as index
                                   columns=['t_'+str(i) for i in range(1,array.shape[1]+1)])

    for col in temp_transcript.columns:
        df_for_metric[col] = temp_transcript[col]
        
    return df_for_metric

def find_std_dev_sub(pred_df, true_df):
    # input: dataframes created by convert'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    temp_pred = pred_df[['gender','race', 'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    temp_true = true_df[['gender','race', 'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    
    pred_prob_mat = temp_pred.groupby(['gender','race']).mean()
    truth_prob_mat = temp_true.groupby(['gender','race']).mean()

    pred_std, truth_std = pred_prob_mat.std().values, truth_prob_mat.std().values
    pred_mean, truth_mean = pred_prob_mat.mean().values, truth_prob_mat.mean().values
    


    return pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat


def find_std_dev_gender_sub(pred_df, true_df):
    # input: dataframes created by convert_dict_to_categorical containing 14 rating category columns
    temp_pred = pred_df[['gender','race', 'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    temp_true = true_df[['gender','race', 'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    
    pred_prob_mat = temp_pred.groupby(['gender']).mean()
    truth_prob_mat = temp_true.groupby(['gender']).mean()

    pred_std, truth_std = pred_prob_mat.std().values, truth_prob_mat.std().values
    pred_mean, truth_mean = pred_prob_mat.mean().values, truth_prob_mat.mean().values
    


    return pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat

def find_std_dev_race_sub(pred_df, true_df):
    # input: dataframes created by convert_dict_to_categorical containing 14 rating category columns
    temp_pred = pred_df[['gender','race', 'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    temp_true = true_df[['gender','race', 'fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']]
    
    pred_prob_mat = temp_pred.groupby(['race']).mean()
    truth_prob_mat = temp_true.groupby(['race']).mean()

    pred_std, truth_std = pred_prob_mat.std().values, truth_prob_mat.std().values
    pred_mean, truth_mean = pred_prob_mat.mean().values, truth_prob_mat.mean().values
    


    return pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat