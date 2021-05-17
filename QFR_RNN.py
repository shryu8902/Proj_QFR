from tensorflow.keras.layers import Input, RepeatVector,Bidirectional,LSTM,TimeDistributed,Dense,Flatten, LeakyReLU,Conv1D, PReLU, Dropout, Concatenate, GRU, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# Add positional embedding a. channelwise, b. just add to original data
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)


def QRNN_model(RN=[256,256], FN=[128,64], Q_bin=[0.1,0.3,0.5,0.7,0.9], dr_rates=0.2, PE=None, PE_d = 16, BN=False):
    inputs = Input(shape =(16) ,name='input')
    num_rows = tf.shape(inputs,name='num_rows')[0]
    inputs_extend = RepeatVector(360,name='extend_inputs')(inputs)

    if PE == 'add':
        pos_enc_tile = tf.tile(positional_encoding(360,16), [num_rows, 1,1],name='pos_enc_tile')
        inputs_extend = 0.5*pos_enc_tile+inputs_extend    
    elif PE == 'concat':
        pos_enc_tile = tf.tile(positional_encoding(360,PE_d), [num_rows, 1,1],name='pos_enc_tile')
        inputs_extend = Concatenate()([inputs_extend,pos_enc_tile])

    if len(RN) !=0:
        for i, n_cell in enumerate(RN):
            if i == 0:
                lstm = Bidirectional(LSTM(n_cell, return_sequences=True,dropout=dr_rates))(inputs_extend)       
                # lstm = LSTM(n_cell, return_sequences=True,dropout=dr_rates)(inputs_extend)       

            else:
                lstm = Bidirectional(LSTM(n_cell, return_sequences=True,dropout=dr_rates))(lstm)    
                # lstm = LSTM(n_cell, return_sequences=True,dropout=dr_rates)(inputs_extend)       

        if len(FN)!=0:
            for i,j in enumerate(FN):
                if i ==0:
                    FN_layer = TimeDistributed(Dense(j,activation=LeakyReLU()))(lstm)
                else:
                    FN_layer = TimeDistributed(Dense(j,activation=LeakyReLU()))(FN_layer) 
            FN_drop = Dropout(dr_rates)(FN_layer)
            
            FN_out = [TimeDistributed(Dense(1),name='out_{}'.format(x))(FN_drop) for x in range(len(Q_bin))]
        else :
            FN_out = [TimeDistributed(Dense(1),name='out_{}'.format(x))(lstm) for x in range(len(Q_bin))]
        outputs = [Flatten(name='flat_out_{}'.format(i))(x) for i,x in enumerate(FN_out)] # update

    else :
        for i, n_neuron in enumerate(FN):
            if i==0:
                FN_layer = Dense(n_neuron,activation=LeakyReLU())(inputs)
            else:
                FN_layer = Dense(n_neuron, activation=LeakyReLU())(FN_layer)
            if BN == True:
                FN_layer = BatchNormalization()(FN_layer) 
        FN_drop = Dropout(dr_rates)(FN_layer)
        outputs = [Dense(360,name='out_{}'.format(x))(FN_drop) for x in range(len(Q_bin))]

    model= Model(inputs, outputs)
    return model
    