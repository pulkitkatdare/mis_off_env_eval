import numpy as np 
import matplotlib.pyplot as plt 
import torch 

import pickle as pkl 
T = 20000
colors = ['#000000', '#00441b', '#7f0000', '#084081', '#4d004b', '#084081', '#6a3d9a', '#ff7f00', '#b15928', '#b15928'] 
algo_type = 'GenDICE'
params_p = -1
params_q = 10.0
real_policy = 0.4 #0.2, 0.4
sim_policy = 0.0 #0.1, 0.2, 0.3

# threshold (0.4) - 0.0: 1.0, 0.1: 0.08, 0.2: 0.02
# threshold (0.5) - 0.0: 1.0, 0.1: 0.08, 0.2: 0.001
# threshold (0.6) - 0.0: 1.0, 0.1: 0.12, 0.2: 0.02


# Target policy = 0.0
# -2 : -3.9222
# -1: -4.532
# 0: -5.10
# 1: -4.60

# Target policy = 0.1
# -2 : -4.11
# -1: -4.555
# 0: -4.58
# 1: -4.36

# Target policy = 0.2
# -2 : -4.36
# -1: -4.555
# 0: -4.36
# 1: -4.85




timesteps = 20000
fig, ax = plt.subplots(1,1, figsize=(16, 6))
for color_index, algo_type in enumerate(['Beta-DICE', 'GenDICE', 'GradientDICE', 'DualDICE']):
    print (algo_type)
    error = np.zeros((10, 401))
    tau = np.zeros((10, 401))
    loss = np.zeros((10, 401))
    a = []
    for index, exp_id in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        file_appender = str(int(10*params_p)) + '_' + str(int(10*params_q)) + str(int(10*real_policy)) + '_' + str(int(10*sim_policy)) + '_' + str(timesteps)
        if algo_type == 'Beta-DICE':
            correction = 'GradientDICE'
        else:
            correction = algo_type
        with open('./dice_log/' + algo_type + '_' + file_appender + '_' + str(exp_id) + '_' + str(correction) + '.pkl', 'rb') as f:
            l = pkl.load(f)
            print (len(l))
        first_time = False
        min_val = 1e6
        for i, li in enumerate(l):
            error[index, i] = np.log10(li[0].detach().cpu().numpy())
            tau[index, i] = li[1]
            loss[index, i] = max(0.5*(li[2])**2, 1e-10)
            if error[index, i] < min_val:
                min_val = error[index, i]
                tau_min = tau[index, i]
            if ((li[1] > 1.0) and (first_time is False)):
                first_time = True
                print (error[index, i-1:i+1])
                a.append(error[index, i]) 
        print ("min", min(error[index, :]), tau_min) #Beta-DICE1.02.03_final.pkl
        if first_time is False:
            a.append(error[index, -1])
            print (error[index, -1])
        #print ('./dice_log/' + algo_type + str(10*sim_policy) + str(10*real_policy) + str(int(10*params_p)) + '_final.pkl')
        with open('./dice_log/' + algo_type + str(10*sim_policy) + str(10*real_policy) + str(int(10*params_p)) + '_final.pkl', 'wb') as f:
            pkl.dump(a, f)
    loss = np.log10(loss + 0.01)
        #print ((error[index, -1]))
        #a.append(((error[index, -1])))
    mean_error = np.mean(error, axis=0)
    std_error = np.std(error, axis=0)
    x = np.arange(len(l))
    #ax[0].plot(mean_error, color=colors[color_index])
    #ax[0].fill_between(x, mean_error-std_error, mean_error+std_error, color=colors[color_index], alpha=0.1)
    mean_tau = np.mean(loss, axis=0)
    std_tau = np.std(loss, axis=0)
    ax.plot(mean_tau, color=colors[color_index])
    ax.fill_between(x,  mean_tau-std_tau, mean_tau+std_tau, color=colors[color_index], alpha=0.1)
    #print (mean_error[-1], std_error[-1])
    print ("mean", np.mean(a))

    
ax.grid(visible=True)
#ax[1].grid(visible=True)
ax.set_xlabel('iterations')
ax.set_ylabel('Log MSE Error')
#ax[1].set_xlabel('iterations')
#ax[1].set_ylabel('Average Tau')
plt.savefig('temp.png')
