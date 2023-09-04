#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ctypes import sizeof
from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
import copy

def encode(states, environment):
    R = np.shape(states)[0]
    out = np.zeros(R)
    for i in range(R):
        taxi_row = states[i, 0]
        taxi_col = states[i, 1]
        pass_loc = states[i, 2]
        dest_idx = states[i, 3]
        out[i] = int(environment.encode(taxi_row, taxi_col, pass_loc, dest_idx))
    return out 

def decode(state, env):
    R = state.size(0)
    out = np.zeros((R, 4))
    for i in range(R):
        decode_state = list(env.decode(state[i].cpu().item()))
        decode_state = np.array(decode_state).reshape(-1)
        out[i, :] = decode_state
    return tensor(out)

class OffPolicyEvaluationDiscrete(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.loss_min = 1e10
        self.use_cuda = config.use_cuda
        self.task = config.task_fn()
        torch.manual_seed(0)
        self.DICENet = config.dice_net_fn()
        self.DICENet_target = config.dice_net_fn()
        self.DICENet_target.load_state_dict(self.DICENet.state_dict())
        if self.use_cuda:
            self.DICENet = config.dice_net_fn().cuda()
            self.DICENet_target = config.dice_net_fn().cuda()
        else:
            self.DICENet = config.dice_net_fn()
            self.DICENet_target = config.dice_net_fn()
        self.network = config.network_fn()
        self.replay_real = config.replay_fn()
        self.replay_sim = config.replay_fn()
        self.total_steps = 0
        self.data_collection_noise = config.data_collection_noise
        self.std = config.sim_policy
        self.env_p = config.environment_p
        self.env_q = config.environment_q
        self.action_dim = 1
        self.action_space = 6
        self.state_dim = 4
        self.convert_to_replay_buffer(pickle.load(open(config.file_p, "rb")), self.replay_real)
        self.convert_to_replay_buffer(pickle.load(open(config.file_q, "rb")), self.replay_sim)
        self.curr_loss = 0.0
        
        #self.load('./data/GradientDICE/%s-policy' % config.game)
        self.with_beta = config.with_beta
        self.beta_network = config.beta_factor
        if self.use_cuda:
            if self.with_beta:
                for i in range(len(self.beta_network)):
                    self.beta_network[i] = self.beta_network[i].cuda()
        
        #self.load('./data/GradientDICE/%s-policy' % config.game)
        
        self.expert_policy = config.expert_policy
        
        self.deterministic = config.deterministic
        self.noise_std = config.noise_std
        self.data_collection_noise = config.data_collection_noise
        self.beta_network = config.beta_factor
        self.loss_history = []
        
        self.oracle_perf = self.load_oracle_perf()
        print('True performance: %s' % (self.oracle_perf))


    def convert_to_replay_buffer(self, pickle_object, replay_buffer):
        
        for experience in pickle_object:
            states, action, next_states, rewards, done = experience 
            input_states = np.reshape(states, (1, self.state_dim))
            input_rewards = np.zeros((1, 1))
            input_rewards[0, 0] = rewards
            input_actions = np.reshape(action, (1, self.action_dim))#
            #input_actions[0, :] = action
            input_next_states = np.reshape(next_states, (1, self.state_dim))
            input_done = np.zeros((1, 1))
            input_done[0, 0] = done
            experiences = list(zip(input_states, input_actions, input_rewards, input_next_states, input_done))
            replay_buffer.feed_batch(experiences)

    def sample_action(self, states, std):
        lstm_states=None
        episode_start = np.ones((1,), dtype=bool)
        states = encode(states.cpu().numpy(), self.env_p.env)
        R = np.shape(states)[0]
        actions = np.zeros(R)
        for i in range(R):
            if np.random.uniform() < std:
                actions[i] = action = np.random.randint(self.action_space)
            else:
                actions[i] = np.argmax(self.expert_policy[int(states[i])])
        return tensor(actions).view(-1, 1)

    def eval_episode(self, environment=None):
        config = self.config
        env = config.eval_env
        state = environment.reset()
        rewards = []
        episode_start = np.ones((1,), dtype=bool)
        lstm_states = None
        timesteps = 0
        while True:
            timesteps += 1
            action = np.argmax(self.expert_policy[state])
            if np.random.rand() < self.noise_std:
                action = np.random.randint(self.action_space)
            
            state, reward, done, info = environment.step(action)
            rewards.append(reward)
            ret = None#info[0]['episodic_return']
            if timesteps > 150:
                #print('Computing true performance: %s' % ret)
                break
        if config.discount == 1:
            return np.mean(rewards)
        ret = 0
        for r in reversed(rewards):
            ret = r + config.discount * ret
        return ret

    def load_oracle_perf(self):
        return self.compute_oracle()

    def compute_oracle(self):
        config = self.config
        print (config.game)
        if config.game in ['Reacher-v2', 'CartPole-v1', 'Acrobot-v1']:
            n_ep = 100
        elif config.game in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Swimmer-v2', 'RoboschoolHalfCheetah-v1']:
            n_ep = 100
        elif config.game in ['Taxi-v3']:
            n_ep = 10
        else:
            raise NotImplementedError
        perf = []
        for ep in range(n_ep):
            perf.append(self.eval_episode(self.env_p))
        if config.discount == 1:
            return np.mean(perf)
        else:
            return (1 - config.discount) * np.mean(perf)

    def step(self):
        config = self.config
        if config.correction == 'no':
            return
        experiences = self.replay_real.sample()
        states, actions, _, next_states, terminals = experiences
        state_action_state = np.concatenate((states, actions), axis=1)
        if self.use_cuda:
            if self.with_beta:
                beta_target = torch.stack([self.beta_network[i].predict(tensor(state_action_state).cuda()) for i in range(len(self.beta_network))])
                beta_target = torch.mean(beta_target, dim=0)
        else:
            if self.with_beta:
                beta_target = ([self.beta_network[i].predict(torch.tensor(state_action_state).type(torch.FloatTensor)) for i in range(len(self.beta_network))])
                beta_target = torch.mean(beta_target, dim=0)
        #beta_target = beta_target/beta_target.sum()
        #print (beta_target)
        
        states = tensor(states)
        actions = tensor(actions)
        next_states = tensor(next_states)
        masks = tensor(1 - terminals).unsqueeze(-1)
        
        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            next_states = next_states.cuda()
            masks = masks.cuda()

        next_actions = self.sample_action(next_states, self.std)
        states_0 = tensor(config.sample_init_states())
        states_0 = decode(states_0, self.env_p.env)
        if self.use_cuda:
            states_0 = states_0.cuda()
        
        actions_0 = self.sample_action(states_0, self.std)
        if self.use_cuda:
            actions = actions.cuda()
            actions_0 = actions_0.cuda()
            next_actions = next_actions.cuda()

        tau = self.DICENet.tau(states, actions)
        f = self.DICENet.f(states, actions)
        f_next = self.DICENet.f(next_states, next_actions)
        f_0 = self.DICENet.f(states_0, actions_0)
        u = self.DICENet.u(states.size(0))

        tau_target = self.DICENet_target.tau(states, actions).detach()
        f_target = self.DICENet_target.f(states, actions).detach()
        f_next_target = self.DICENet_target.f(next_states, next_actions).detach()
        f_0_target = self.DICENet_target.f(states_0, actions_0).detach()
        u_target = self.DICENet_target.u(states.size(0)).detach()

        experiences = self.replay_sim.sample()
        states, actions, _, next_states, terminals = experiences

        states = tensor(states)
        actions = tensor(actions)
        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()

        tau_real = self.DICENet.tau(states, actions)
        tau_real_target = self.DICENet.tau(states, actions).detach()
        f_real = self.DICENet.f(states, actions)
        f_real_target = self.DICENet.f(states, actions).detach()



        if config.correction == 'GenDICE':
            if self.with_beta:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next)*beta_target - \
                        tau_real_target * (f_real + 0.25 * f_real.pow(2)) + config.lam * (u * tau_real_target - u - 0.5 * u.pow(2))
                J_convex = (1 - config.discount) * f_0_target + (config.discount * tau * f_next_target)*beta_target - \
                       tau_real * (f_real_target + 0.25 * f_real_target.pow(2)) + \
                       config.lam * (u_target * tau_real - u_target - 0.5 * u_target.pow(2))
            else:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next - \
                    tau_target * (f + 0.25 * f.pow(2)) + config.lam * (u * tau_target - u - 0.5 * u.pow(2)))
                J_convex = (1 - config.discount) * f_0_target + config.discount * tau * f_next_target - \
                    tau * (f_target + 0.25 * f_target.pow(2)) + \
                       config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2))

        # Only see here for optimization
        elif config.correction == 'GradientDICE':
            if self.with_beta:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next)*beta_target - \
                            tau_real_target * f_real - 0.5 * f_real.pow(2) + config.lam * (u * tau_real_target - u - 0.5 * u.pow(2))
                J_convex = (1 - config.discount) * f_0_target + (config.discount * tau * f_next_target)*beta_target - \
                        tau_real * f_real_target - 0.5 * f_real_target.pow(2) + \
                        config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2))
        ### only see here ...
            else:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next - \
                            tau_target * f - 0.5 * f.pow(2) + config.lam * (u * tau_target - u - 0.5 * u.pow(2)))
                J_convex = (1 - config.discount) * f_0_target + (config.discount * tau * f_next_target - \
                        tau * f_target - 0.5 * f_target.pow(2) + \
                        config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2)))
                
        elif config.correction == 'DualDICE':
            if self.with_beta:
                J_concave = ((f_target - config.discount * f_next_target) * tau - tau.pow(3).mul(1.0 / 3))*beta_target   \
                            - (1 - config.discount) * f_0_target
                J_convex = ((f - config.discount * f_next) * tau_target - tau_target.pow(3).mul(1.0 / 3))*beta_target - \
                        (1 - config.discount) * f_0
            else:
                J_concave = ((f_target - config.discount * f_next_target) * tau - tau.pow(3).mul(1.0 / 3))   \
                            - (1 - config.discount) * f_0_target
                J_convex = ((f - config.discount * f_next) * tau_target - tau_target.pow(3).mul(1.0 / 3)) - \
                        (1 - config.discount) * f_0
            
                #*beta_target 
        else:
            raise NotImplementedError

        loss = (J_convex - J_concave) * masks
        self.DICENet.opt.zero_grad()
        loss.mean().backward()
        if loss.mean().detach() < self.loss_min:
            self.loss_min = loss.mean().detach()
            self.best_model_state_dict = self.DICENet.state_dict()
        self.DICENet.opt.step()
        if self.total_steps % config.target_network_update_freq == 0:
            self.DICENet_target.load_state_dict(self.DICENet.state_dict())
        self.curr_loss = J_convex.mean().detach()

        self.total_steps += 1


    def eval_episodes(self):
        if self.with_beta is False:
            experiences = self.replay_real.sample(1000)#len(self.replay.data))
        else:
            experiences = self.replay_sim.sample(1000)#len(self.replay.data))
        states, actions, rewards, next_states, terminals = experiences
        
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
        if self.config.correction == 'no':
            tau = 1
        else:
            tau = self.DICENet.tau(states, actions)
        perf = (tau * rewards).mean()
        loss = (perf - self.oracle_perf).pow(2).mul(0.5)
        print (loss.item(), tau.mean().item())
        self.loss_history.append((loss, tau.mean().item(), self.curr_loss))

