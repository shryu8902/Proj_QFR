from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout, Concatenate, GRU, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# Add positional embedding a. channelwise, b. just add to original data
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model)) 
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # (p,1) > pos
                            np.arange(d_model)[np.newaxis, :],   # (1,d) > angle_rates
                            d_model)                             # (p,d) = angle_rads
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]  # to make (N,w,d), actually (1,w,d)
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)


def QRNN_model(RN=[256,256], FN=[128,64], Q_bin=[0.1,0.3,0.5,0.7,0.9], dr_rates=0.2, PE=None, PE_d = 16, BN=False):
    # RN : list of lstm cell
    # FN : list of neurons for dense layer
    # Q_bin : list of target quantiles
    # dr_rates = dropout rates
    # PE : indicator for using PE
    # PE_d : dimension of PE vector
    # BN : indicator for using Batch normalization
    inputs = Input(shape =(16) ,name='input')   #Batch size is not specified. shape = 16 >> input data dimension is None x 16
    num_rows = tf.shape(inputs,name='num_rows')[0] #mini batch size
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs) # None x 16 >> None x 360 x 16

    if PE == 'add':
        pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile') # N x 360 x 16
        inputs_extend = 0.5*pos_enc_tile+inputs_extend    # None(N) x 360 x 16
    elif PE == 'concat':
        pos_enc_tile = tf.tile(positional_encoding(360,PE_d), [num_rows, 1,1],name='pos_enc_tile') # N x 360 x d
        inputs_extend = Concatenate()([inputs_extend,pos_enc_tile]) # None(N) x 360 x (16+d)

    if len(RN) !=0:         # i.e., if using LSTM
        for i, n_cell in enumerate(RN): 
            if i == 0:
                lstm = Bidirectional(LSTM(n_cell, return_sequences=True,dropout=dr_rates))(inputs_extend)       
                # lstm = LSTM(n_cell, return_sequences=True,dropout=dr_rates)(inputs_extend)       
            else:
                lstm = Bidirectional(LSTM(n_cell, return_sequences=True,dropout=dr_rates))(lstm)    
                # lstm = LSTM(n_cell, return_sequences=True,dropout=dr_rates)(inputs_extend)       
        if len(FN)!=0:  # stacking dense (linear, fully-connected) layers
            for i,j in enumerate(FN):
                if i ==0:
                    FN_layer = TimeDistributed(Dense(j,activation=LeakyReLU()))(lstm) # connects first dense layer to below LSTM layers
                else:
                    FN_layer = TimeDistributed(Dense(j,activation=LeakyReLU()))(FN_layer)  # other wise, connect to pervious dense layers
            FN_drop = Dropout(dr_rates)(FN_layer)   #Apply dropout for final dense layer
            
            FN_out = [TimeDistributed(Dense(1),name='out_{}'.format(x))(FN_drop) for x in range(len(Q_bin))] # From FN_layer, get Q outputs >> [[N x W x 1],...[N x W x 1]]
        else :
            FN_out = [TimeDistributed(Dense(1),name='out_{}'.format(x))(lstm) for x in range(len(Q_bin))] # If dense layers are not used, directly output Qs from LSTM
        outputs = [Flatten(name='flat_out_{}'.format(i))(x) for i,x in enumerate(FN_out)] # Flatten output [N x W x 1] to [N x W]

    else :      # i.e., without LSTM, just a multi layer perceptron.
        for i, n_neuron in enumerate(FN):
            if i==0:
                FN_layer = Dense(n_neuron,activation=LeakyReLU())(inputs)  # not inputs extend, thus N x 16 >> N x d'
            else:
                FN_layer = Dense(n_neuron, activation=LeakyReLU())(FN_layer)
            if BN == True:
                FN_layer = BatchNormalization()(FN_layer)   # Apply batch normalization
        FN_drop = Dropout(dr_rates)(FN_layer)
        outputs = [Dense(360,name='out_{}'.format(x))(FN_drop) for x in range(len(Q_bin))] # N x d' >> N x 360  == N x W, thus output is list of 

    model= Model(inputs, outputs)
    return model
    