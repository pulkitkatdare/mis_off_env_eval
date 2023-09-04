from models.BetaNet import BetaNetwork
import pickle 
from torch.utils.data import DataLoader
from tqdm import tqdm
from oee_dataset import BetaEstimationDataset

def calculate_beta(env, log, file_p, file_q,
                num_epochs = 5,
                batch_size = 64,
                learning_rate = 1e-4,
                l2_regularization = 0.01,
                use_cuda = True):

    if env == 'RoboschoolHalfCheetah-v1':
        dataset = BetaEstimationDataset(filename_p=file_p, filename_q=file_q, action_space=6)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        beta_network = BetaNetwork(state_dim=32, learning_rate=learning_rate, tau=l2_regularization, seed=1234, action_dim = 6)
        if use_cuda:
            beta_network = beta_network.to('cuda:0')

    if env == 'CartPole-v1':
        dataset = BetaEstimationDataset(filename_p=file_p, filename_q=file_q, action_space=1)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        beta_network = BetaNetwork(state_dim=5, learning_rate=learning_rate, tau=l2_regularization, seed=1234, action_dim = 1)
        if use_cuda:
            beta_network = beta_network.to('cuda:0')

    
    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        for iteration, data in enumerate(tqdm(dataloader)):
            data_p = data[0]
            data_q = data[1]
            if use_cuda:
                data_p = data_p.to('cuda:0')
                data_q = data_q.to('cuda:0')
            
            loss = beta_network.train_step(states_p=data_p, states_q=data_q)
            epoch_losses.append(loss.item())
            
            if (iteration%100 == 0):
                print ("loss:", loss.item())
                with open(log + '/epoch_loss_' + str(epoch) + '.pkl', 'wb') as fp:
                    pickle.dump(epoch_losses, fp,  protocol=pickle.HIGHEST_PROTOCOL)
        #torch.save(beta_network.state_dict(), log + '/beta_model_' + file_appender + '_' + str(epoch) + '.ptr')
    return beta_network