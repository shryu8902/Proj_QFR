#%%
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow_addons as tfa   # for calculating pinball loss
import gc, pickle, wandb, setGPU  
from wandb.keras import WandbCallback   # for model saving and management

import numpy as np
import pandas as pd
import tqdm, glob, pickle, datetime, re, time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split        # train, validation, test sets
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # training scalers
    
import random as python_random      # for reproducibility

import QFR_RNN

def ensure_dir(file_path):  # if there is no directory then create one
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def MAPE(y_true,y_pred):    # MAPE with mean along channel i.e., N x W >> N 
    ape = abs(y_true-y_pred)/y_true*100
    mape = np.mean(ape,axis=-1)
    return mape
def MSE(y_true,y_pred):    # MSE with mean along channel i.e., N x W >> N
    se = (y_true-y_pred)**2
    mse = np.mean(se,axis=-1)
    return mse

def RandomSetting(SEED):   # Random setting with specific SEED
    np.random.seed(SEED)    # numpy randomness
    python_random.seed(SEED)    # python randomness
    tf.random.set_seed(SEED)    # tensorflow randomness

def ErrorCalculator(in_hat, in_real, out_hat, out_real):    # reconstruction of test_in, test_out. 
    # calculate overall mean error
    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat)),3)  # N >> 1
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat)),3)
    return mape_in, mape_out, mse_in, mse_out

def ErrorCalculator2(in_hat, in_real, out_hat, out_real):
    # calculate overall mean error + standard deviation
    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_in_std = np.round(np.std(MAPE(y_true=in_real, y_pred=in_hat)),3)
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mape_out_std = np.round(np.std(MAPE(y_true=out_real,y_pred=out_hat)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_in_std = np.round(np.std(MSE(y_true=in_real,y_pred=in_hat)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat)),3)
    mse_out_std = np.round(np.std(MSE(y_true=out_real,y_pred=out_hat)),3)

    return mape_in,mape_in_std, mape_out,mape_out_std, mse_in, mse_in_std, mse_out, mse_out_std

def ErrorCalculator_SG2(in_hat, in_real, out_hat, out_real):
    # optional
    in_hat2=in_hat.copy()
    out_hat2=out_hat.copy()
    in_hat2[in_real<=1.0332]=in_real[in_real<=1.0332]
    out_hat2[out_real<=1.0332]=out_real[out_real<=1.0332]

    mape_in = np.round(np.mean(MAPE(y_true=in_real, y_pred=in_hat2)),3)
    mape_in_std = np.round(np.std(MAPE(y_true=in_real, y_pred=in_hat2)),3)
    mape_out = np.round(np.mean(MAPE(y_true=out_real,y_pred=out_hat2)),3)
    mape_out_std = np.round(np.std(MAPE(y_true=out_real,y_pred=out_hat2)),3)
    mse_in = np.round(np.mean(MSE(y_true=in_real,y_pred=in_hat2)),3)
    mse_in_std = np.round(np.std(MSE(y_true=in_real,y_pred=in_hat2)),3)
    mse_out = np.round(np.mean(MSE(y_true=out_real,y_pred=out_hat2)),3)
    mse_out_std = np.round(np.std(MSE(y_true=out_real,y_pred=out_hat2)),3)

    return mape_in,mape_in_std, mape_out,mape_out_std, mse_in, mse_in_std, mse_out, mse_out_std


#%%
# Below parameters are related to wandb library
PROJ_NAME = 'Proto'
VER_NAME = 'RNN' # select one among FNN, RNN, RNN_aPE, RNN_cPE, QRNN_cPE, the name of model
DESC = 'simple RNN model without PE and Q for SG1' # description for model
Q_LIST = [0.5] #[0.5] or [0.1,0.3,0.5,0.7,0.9] # list of target quantiles.

# below is the dictionary for model configuration
config ={
        "opt": 'adam', 
        "Q_list" : Q_LIST,
        "RNN_structure":[128], # Choose between FNN : [] RNN :[256,256]
        "dense_structure" : [], # Choose between FNN : [64, 128, 256], SFNN [512], DFNN [64,128,256,512] RNN : [128,64]
        "dropout_rates" : 0.2,
        "positional_encoding_type" : None, # Choose between None, 'concat', 'add'
        "encoding_dimension" : 0,
        "vername":VER_NAME,
        "epochs": 150,
        "batch" : 512,  # size of minibatch
        "loss" : 'mse' if len(Q_LIST) ==1 else 'pinball_losses', # If target Qs are given, use pinball oss
        "desc" : DESC,
        "BN" : False
        }
#%%
# Set train/val/test_in/test_out datasets Number : (60637/ 7988/ 18000 / 18000) // Percentage (58 / 8 / 17 / 17) 
# test_out is fixed to 18000.
# size of test_in is selected to equal to that of test_out. 
val_ind,test_ind = train_test_split(range(25988),test_size=18000, random_state=0)

train_ = np.load('/home/ashryu/Proj_FR/DATA/train_input.npy') # 60637 
val_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[val_ind,...] # 7988

train_input_ = train_[:,1:] # First column may have index
val_input_ = val_[:,1:]

# train_output_ = np.load('./Data/train_output.npy')
# val_output_ = np.load('./Data/test_output.npy')[val_ind,...]
train_output_ = np.load('/home/ashryu/Proj_FR/DATA/train_output.npy')
val_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[val_ind,...]

with open('/RAID8T/ashryu/Proj_FR/DATA/new_IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('/RAID8T/ashryu/Proj_FR/DATA/new_OUT_SCALER.pickle','rb') as f:
# with open('./Data/OUT_SCALER_SG1.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)

SCALE = 'standard'
train_input = IN_SCALER[SCALE].transform(train_input_)  # train_input_ : N x 16
val_input = IN_SCALER[SCALE].transform(val_input_) # val_input_ : N x 16
train_output = OUT_SCALER[SCALE].transform(train_output_.reshape(-1,1)).reshape(-1,360) # N x W >> NW x 1 >> N x W
val_output = OUT_SCALER[SCALE].transform(val_output_.reshape(-1,1)).reshape(-1,360) # N x W >> NW x 1 >> N x W

#%%
for SEED in range(5):
    vername = VER_NAME+'/S{}'.format(SEED) 
    run = wandb.init(project=PROJ_NAME,config=config)
    run.name= VER_NAME+'_S{}'.format(SEED)+'_'+run.id
    cfg = wandb.config
    cfg.update({'SEED':SEED})
    RandomSetting(SEED)
    
    ensure_dir('./Model/{}/{}/'.format(PROJ_NAME, vername))
    path = './Model/{}/{}'.format(PROJ_NAME, vername)+'/e{epoch:04d}.ckpt'

    base_model = QFR_RNN.QRNN_model(
                RN=cfg.RNN_structure,
                FN=cfg.dense_structure,
                Q_bin=cfg.Q_list,
                dr_rates=cfg.dropout_rates,
                PE=cfg.positional_encoding_type,
                PE_d=cfg.encoding_dimension,
                BN=cfg.BN)

    if len(cfg.Q_list)>1:
        losses_list = [tfa.losses.PinballLoss(tau=tau) for tau in cfg.Q_list]
    else :
        losses_list = 'mse'
    
    base_model.compile(optimizer=cfg.opt, loss = losses_list, metrics=['mean_squared_error'])

    checkpoint = ModelCheckpoint(path, monitor = 'val_loss',verbose = 1,
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-5)

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    hist = base_model.fit(x=train_input, y=[train_output for x in range(len(cfg.Q_list))],
                validation_data=(val_input,[val_output for x in range(len(cfg.Q_list))]),
                callbacks=[checkpoint, reduce_lr, early_stopping, WandbCallback()],
                epochs=cfg.epochs,
                batch_size=cfg.batch)
    run.finish()
    K.clear_session()
