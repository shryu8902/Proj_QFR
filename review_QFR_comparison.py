#%%
val_ind,test_ind = train_test_split(range(25988),test_size=18000, random_state=0)

test_in_ = np.load('/home/ashryu/Proj_FR/DATA/test_input.npy')[test_ind,...]
test_out_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_input.npy')
test_in_input_ = test_in_[:,1:]
test_out_input_ = test_out_[:,1:]
test_in_output_ = np.load('/home/ashryu/Proj_FR/DATA/test_output.npy')[test_ind,...]
test_out_output_ = np.load('/home/ashryu/Proj_FR/DATA/untrain_output.npy')

with open('/RAID8T/ashryu/Proj_FR/DATA/new_IN_SCALER.pickle','rb') as f:
    IN_SCALER = pickle.load(f)
with open('/RAID8T/ashryu/Proj_FR/DATA/new_OUT_SCALER.pickle','rb') as f:
    OUT_SCALER = pickle.load(f)
SCALE = 'standard'

test_in_input = IN_SCALER[SCALE].transform(test_in_input_)
test_out_input = IN_SCALER[SCALE].transform(test_out_input_)
test_in_output = OUT_SCALER[SCALE].transform(test_in_output_.reshape(-1,1)).reshape(-1,360)
test_out_output = OUT_SCALER[SCALE].transform(test_out_output_.reshape(-1,1)).reshape(-1,360)

#%%
result=[]
plot_data=[]
api = wandb.Api()

exp_list = pd.read_csv('./Model/all_wandb_list.csv')
#%%
VER_NAME='QRNN_cpe8'
id_list = exp_list.iloc[[VER_NAME+'_S' in x for x in exp_list.Name]].Name
id_list

#%%
# Model : FNN, RNN, RNN_cpe8, RNN_aPE,
ensem_hat_in=[]
ensem_hat_out=[]

for info in tqdm.tqdm(id_list):
    run_id = info.split('_')[3]
    cfg = api.run("shryu8902/QFR/{}".format(run_id)).config
    base_model = QFR_RNN.QRNN_model(
                RN=cfg['RNN_structure'],
                FN=cfg['dense_structure'],
                Q_bin=cfg['Q_list'],
                dr_rates=cfg['dropout_rates'],
                PE=cfg['positional_encoding_type'],
                PE_d=cfg['encoding_dimension'],
                BN=False) 
    # load weights
    # case for RNN
    if (VER_NAME=='RNN') and (cfg['SEED']==0):  
        base_model.load_weights('./Model/{}/S{}/e0141.ckpt'.format(VER_NAME,cfg['SEED']))
    else:
        latest = tf.train.latest_checkpoint('./Model/{}/S{}'.format(VER_NAME, cfg['SEED']))
        base_model.load_weights(latest)
    # inference
    test_in_output_hat = base_model.predict(test_in_input,batch_size=128)[2]
    test_out_output_hat = base_model.predict(test_out_input,batch_size=128)[2]
    test_in_output_hat_ = OUT_SCALER[SCALE].inverse_transform(test_in_output_hat.reshape(-1,1)).reshape(-1,360)
    test_out_output_hat_ = OUT_SCALER[SCALE].inverse_transform(test_out_output_hat.reshape(-1,1)).reshape(-1,360)        
    # calc errors
    mape_in, mape_out, mse_in, mse_out = ErrorCalculator(in_hat=test_in_output_hat_,
                                                        in_real=test_in_output_,
                                                        out_hat=test_out_output_hat_,
                                                        out_real=test_out_output_)

    result.append({'Model':VER_NAME,'SEED':cfg['SEED'],'MAPE_in':mape_in,'MAPE_out':mape_out,
                        'MSE_in':mse_in,'MSE_out':mse_out})

    ensem_hat_in.append(test_in_output_hat_)
    ensem_hat_out.append(test_out_output_hat_)
    plot_data.append({'Model':VER_NAME, 'SEED':cfg['SEED'],'pred_in':test_in_output_hat_,'pred_out':test_out_output_hat_})
# ensemble mean
ensem_in_hat_=np.mean(ensem_hat_in,axis=0)
ensem_out_hat_=np.mean(ensem_hat_out,axis=0)

mape_in, mape_out, mse_in, mse_out = ErrorCalculator(in_hat = ensem_in_hat_,
                                                    in_real = test_in_output_,
                                                    out_hat = ensem_out_hat_,
                                                    out_real = test_out_output_)
result.append({'Model':VER_NAME,'SEED':10,'MAPE_in':mape_in,'MAPE_out':mape_out,
                    'MSE_in':mse_in,'MSE_out':mse_out})
plot_data.append({'Model':VER_NAME, 'SEED':10,'pred_in':ensem_in_hat_,'pred_out':ensem_out_hat_})


#%%
df_result = pd.DataFrame(result,columns=['Model','SEED','MAPE_in','MAPE_out','MSE_in','MSE_out'])
df_result.to_csv('result3.csv')

with open('./Data/plot_data.pickle','wb') as f:
    pickle.dump(plot_data,f)
#%%    
# BElow for quantile graph
VER_NAME='QRNN_cpe8'
id_list = exp_list.iloc[[VER_NAME+'_S' in x for x in exp_list.Name]].Name
id_list

#%%
# Model : FNN, RNN, RNN_cpe8, RNN_aPE,
ensem_hat_in=[[],[],[],[],[]]
ensem_hat_out=[[],[],[],[],[]]
q_plot_data = [[],[],[],[],[]]
for info in tqdm.tqdm(id_list):
    run_id = info.split('_')[3]
    cfg = api.run("shryu8902/QFR/{}".format(run_id)).config
    base_model = QFR_RNN.QRNN_model(
                RN=cfg['RNN_structure'],
                FN=cfg['dense_structure'],
                Q_bin=cfg['Q_list'],
                dr_rates=cfg['dropout_rates'],
                PE=cfg['positional_encoding_type'],
                PE_d=cfg['encoding_dimension'],
                BN=False) 
    # load weights
    # case for RNN
    if (VER_NAME=='RNN') and (cfg['SEED']==0):  
        base_model.load_weights('./Model/{}/S{}/e0141.ckpt'.format(VER_NAME,cfg['SEED']))
    else:
        latest = tf.train.latest_checkpoint('./Model/{}/S{}'.format(VER_NAME, cfg['SEED']))
        base_model.load_weights(latest)
    # inference
    test_in_output_hat = base_model.predict(test_in_input,batch_size=128)
    test_out_output_hat = base_model.predict(test_out_input,batch_size=128)
    for k in range(5):
        denorm_temp_in = OUT_SCALER[SCALE].inverse_transform(test_in_output_hat[k].reshape(-1,1)).reshape(-1,360)
        denorm_temp_out = OUT_SCALER[SCALE].inverse_transform(test_out_output_hat[k].reshape(-1,1)).reshape(-1,360)
        ensem_hat_in[k].append(denorm_temp_in)
        ensem_hat_out[k].append(denorm_temp_out)
        q_plot_data[k].append({'Model':VER_NAME, 'SEED':cfg['SEED'],'pred_in':denorm_temp_in,'pred_out':denorm_temp_out})

    # calc errors
    # mape_in, mape_out, mse_in, mse_out = ErrorCalculator(in_hat=test_in_output_hat_,
    #                                                     in_real=test_in_output_,
    #                                                     out_hat=test_out_output_hat_,
    #                                                     out_real=test_out_output_)

    # result.append({'Model':VER_NAME,'SEED':cfg['SEED'],'MAPE_in':mape_in,'MAPE_out':mape_out,
    #                     'MSE_in':mse_in,'MSE_out':mse_out})

# ensemble mean
for k in range(5):
    ensem_in_hat_=np.mean(ensem_hat_in[k],axis=0)
    ensem_out_hat_=np.mean(ensem_hat_out[k],axis=0)
    q_plot_data[k].append({'Model':VER_NAME, 'SEED':10,'pred_in':ensem_in_hat_,'pred_out':ensem_out_hat_})
#%%
with open('./Data/q_plot_data.pickle','wb') as f:
    pickle.dump(q_plot_data,f)


