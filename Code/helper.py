import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import stats
from scipy.stats import sem
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score
import pickle
import pandas as pd
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.ticker import StrMethodFormatter
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import pandas as pd
import os


rating_names = ['fascinating', 'ingenious', 'jaw-dropping', 'longwinded', 'unconvincing', 'ok']




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


def create_div(res,k):
    div_dic = {}
    for key in res.keys():
        B = np.dot(res[key],res[key].T)
        x = (np.prod(sorted(np.linalg.eigvals(B))[-k:])).real #/ np.prod(sorted(1+np.linalg.eigvals(B))).real
        
        div_dic[key] = x
   
    return div_dic

def prec(l):
    return [float(str(x)[:3]) for x in l]


def do_binning(dl,rc,num_bin):
    #creating zipped list and removing outlier
    rat_cur = sorted(list(zip(dl,rc)))#[lower_limit:upper_limit]
       
    h,e = np.histogram(rat_cur,bins=num_bin)
    div, rating = list(zip(*rat_cur))
    n = len(div)
    min_div, max_div = min(div), max(div)
    div_ar, rating_ar = [[] for x in range(num_bin)], [[] for x in range(num_bin)]
    division = np.linspace(min_div,max_div,num_bin)
    cur_next = 1
    for i in range(n):
        if div[i] <= division[cur_next]:
            div_ar[cur_next-1].append(div[i])
            rating_ar[cur_next-1].append(rating[i])
        else:
            cur_next = cur_next+1
   
    new_div_ar, new_rating_ar = [], []
    for k in range(num_bin):
        if div_ar[k] and rating_ar[k]:
            new_div_ar.append(np.array(div_ar[k]))
            new_rating_ar.append(np.array(rating_ar[k]))
            
            
    mean_divs, mean_rates, std_divs, std_rates = [divs.mean() for divs in new_div_ar], [rates.mean() for rates in new_rating_ar], [sem(divs) for divs in new_div_ar], [sem(rates) for rates in new_rating_ar]#[divs.std() for divs in new_div_ar], [rates.std() for rates in new_rating_ar]
    return np.array(mean_divs), np.array(mean_rates), np.array(std_divs), np.array(std_rates)



def generate_errorbardata(group1_df, group2_df,rat,div_type):
    
    num_rep = 100
    num_cat_div = 5
    bin_rate = "bin_"+rat
    mean_rate_g1, std_rate_g1, mean_rate_g2, std_rate_g2 = [], [], [], []
    for cat in range(num_cat_div):
        cur_group1_df, cur_group2_df = group1_df[group1_df[div_type]==cat], group2_df[group2_df[div_type]==cat]
        group1_sample, group2_sample = [], []
        for rep in range(num_rep):
            
            group1_samp = cur_group1_df.sample(cur_group1_df.shape[0],replace=True)
            group1_sample.append(group1_samp[group1_samp[bin_rate]==1].shape[0]/float(cur_group1_df.shape[0]))

            group2_samp = cur_group2_df.sample(cur_group2_df.shape[0],replace=True)
            group2_sample.append(group2_samp[group2_samp[bin_rate]==1].shape[0]/float(cur_group2_df.shape[0]))

        mean_rate_g1.append(np.mean(group1_sample))
        mean_rate_g2.append(np.mean(group2_sample))

        std_rate_g1.append(sem(group1_sample))
        std_rate_g2.append(sem(group2_sample))
            
    return mean_rate_g1, std_rate_g1, mean_rate_g2, std_rate_g2


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



# Function to compute the SPD Metrics w.r.t. race

def fairness_list_race(df_predict_with_dl,df_predict_only,df_true,rating):

    _, _, _, _, pred_prob_mat_with_dl,truth_prob_mat = find_std_dev_race_sub(df_predict_with_dl, df_true)
    _, _, _, _, pred_prob_mat,truth_prob_mat = find_std_dev_race_sub(df_predict_only, df_true)

    #print(pred_prob_mat_with_dl)
    spd_pred_dl,di_pred_dl,spd_pred,di_pred,spd_true,di_true = [], [], [], [],[], []

    for k,label in enumerate(rating):


        predicted_probs = pred_prob_mat[[label]].values
        true_probs = truth_prob_mat[[label]].values
        predicted_probs_with_dl = pred_prob_mat_with_dl[[label]].values

        #print(label,predicted_probs)
        num_groups = true_probs.shape[0]
        #print(num_groups,"num of groups")
        predicted_probs,true_probs,predicted_probs_with_dl  = predicted_probs.reshape(num_groups,), true_probs.reshape(num_groups,), predicted_probs_with_dl.reshape(num_groups,)

        #Statistical Parity difference
        spd_pred_dl.append(abs(predicted_probs_with_dl[1]-predicted_probs_with_dl[3]))
        spd_pred.append(abs(predicted_probs[1]-predicted_probs[3]))
        spd_true.append(abs(true_probs[1]-true_probs[3]))

        #Disparate Impact
        di_pred_dl.append(abs(1-predicted_probs_with_dl[1]/(predicted_probs_with_dl[3]+0.0001)))
        di_pred.append(abs(1-predicted_probs[1]/(predicted_probs[3]+0.0001)))
        di_true.append(abs(1-true_probs[1]/(true_probs[3]+0.0001)))


    return spd_pred_dl,di_pred_dl,spd_pred,di_pred,spd_true,di_true

# Function to compute the SPD Metrics w.r.t. gender

def fairness_list_gender(df_predict_with_dl,df_predict_only,df_true,rating):

    _, _, _, _, pred_prob_mat_with_dl,truth_prob_mat = find_std_dev_gender_sub(df_predict_with_dl, df_true)
    _, _, _, _, pred_prob_mat,truth_prob_mat = find_std_dev_gender_sub(df_predict_only, df_true)

    #print(pred_prob_mat_with_dl[['gender']].values)
    spd_pred_dl,di_pred_dl,spd_pred,di_pred,spd_true,di_true = [], [], [], [],[], []

    for k,label in enumerate(rating):

        predicted_probs = pred_prob_mat[[label]].values
        true_probs = truth_prob_mat[[label]].values
        predicted_probs_with_dl = pred_prob_mat_with_dl[[label]].values

        #print(label,predicted_probs)
        num_groups = true_probs.shape[0]
        #print(num_groups,"num of groups")
        predicted_probs,true_probs,predicted_probs_with_dl  = predicted_probs.reshape(num_groups,), true_probs.reshape(num_groups,), predicted_probs_with_dl.reshape(num_groups,)

        #Statistical Parity difference
        spd_pred_dl.append(abs(predicted_probs_with_dl[0]-predicted_probs_with_dl[2]))
        spd_pred.append(abs(predicted_probs[0]-predicted_probs[2]))
        spd_true.append(abs(true_probs[0]-true_probs[2]))

        #Disparate Impact
        di_pred_dl.append(abs(1-predicted_probs_with_dl[0]/(predicted_probs_with_dl[2]+0.0001)))
        di_pred.append(abs(1-predicted_probs[0]/(predicted_probs[2]+0.0001)))
        di_true.append(abs(1-true_probs[0]/(true_probs[2]+0.0001)))


    return spd_pred_dl,di_pred_dl,spd_pred,di_pred,spd_true,di_true


# Function to plot the SPD Metrics

def plot_SPD(pred,pred_both,true,rating,metric='SPD',sens='race'):

	rat_ind = [rating_names.index(x) for x in rating]
	pred,pred_both,true = [pred[x] for x in rat_ind], [pred_both[x] for x in rat_ind], [true[x] for x in rat_ind]
	
	xaxis = np.arange(len(rating))
	width = 0.2

	fig, ax = plt.subplots()
	rects1 = ax.bar(xaxis,true  , width,label= 'True', color='#2E473C')
	rects2 = ax.bar(xaxis + width, pred, width, label= 'Without HEM', color='#D9B98D')    
	rects3 = ax.bar(xaxis + width+ width, pred_both , width, label= 'With HEM',color='#6F485E')
	# rects4 = ax.bar(xaxis + width+width+width, pred, width, label='Without HEM',color='#6F485E')

	ax.set_xticks(xaxis+width/2)
	ax.set_xticklabels(rating,fontsize=10,rotation=90)

	plt.legend()

	plt.yticks(fontsize=14)
	plt.title(metric)
	plt.tight_layout() 
	plt.savefig('../Plots/'+'all_'+metric+'_'+sens+'.pdf')
	plt.close()

# Function to plot the CV_prob

def plot_cv(df_predict,df_true):

    pred_std, truth_std, pred_mean, truth_mean, pred_prob_mat,truth_prob_mat = find_std_dev_sub(df_predict, df_true)
    


    pred_mean = pred_mean + 0.00001
    truth_mean = truth_mean + 0.00001
    pred_cv = pred_std / pred_mean
    truth_cv = truth_std / truth_mean

    print(pred_cv, truth_cv)
    marker_list = ['x','o','*','D','+','s']
    colors = ['b', 'c', 'y', 'm', 'r','g']
    for i in range(len(pred_cv)):
        plt.plot([truth_cv[i]],[pred_cv[i]],c=colors[i],marker= marker_list[i],label=rating_names_sub[i])

    #plt.scatter(truth_cv,pred_cv)
    min_range = min(min(pred_std),min(truth_std))
    max_range = max(max(pred_std),max(truth_std))
    #line =np.linspace(min_range,max_range,100)
    line =np.linspace(0,0.9,100)
    plt.ylabel('CV of the predicted label')
    plt.xlabel('CV of the true label')
    plt.plot(line,line)
    plt.legend()
    #plt.subplot(2,2,3)

    plt.savefig('../Plots/CV_prob/cv_prob.pdf')
    plt.close()