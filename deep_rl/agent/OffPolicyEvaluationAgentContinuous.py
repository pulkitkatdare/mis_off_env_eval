#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
import copy
import pickle 
from torch.utils.data import DataLoader




class OffPolicyEvaluationContinuous(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.use_cuda = config.use_cuda
        self.curr_loss = 0.0
        if self.use_cuda:
            self.DICENet = config.dice_net_fn().cuda()
            self.DICENet_target = config.dice_net_fn().cuda()
        else:
            self.DICENet = config.dice_net_fn()
            self.DICENet_target = config.dice_net_fn()
        self.DICENet_target.load_state_dict(self.DICENet.state_dict())
        self.replay_real = config.replay_fn()
        self.replay_sim = config.replay_fn()
        self.total_steps = 0
        self.with_beta = config.with_beta
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.convert_to_replay_buffer(pickle.load(open(config.file_p, "rb")), self.replay_real)
        self.convert_to_replay_buffer(pickle.load(open(config.file_q, "rb")), self.replay_sim)
        
        #self.load('./data/GradientDICE/%s-policy' % config.game)
        self.beta_network = config.beta_factor
        if self.use_cuda:
            for i in range(5):
                self.beta_network[i] = self.beta_network[i].cuda()

        self.loss_history = []
        self.min_loss = 1e6
        self.real_env = config.environment_p
        self.sim_env = config.environment_q
        self.deterministic = False
        self.expert_policy = config.expert_policy
        self.std = config.sim_policy
        self.loss_min = 1e10
        
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
        actions, lstm_states = self.expert_policy.predict(
                states.cpu(),
                state=lstm_states,
                episode_start=episode_start,
                deterministic=self.deterministic)
        actions = torch.from_numpy(actions).type(torch.FloatTensor)
        actions += torch.randn(actions.size()) * std
        return actions

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
            action, lstm_states = self.expert_policy.predict(
                state,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=self.deterministic)
            action += np.random.normal(self.action_dim)*self.std
            state, reward, done, info = environment.step(action)
            rewards.append(reward)
            ret = None#info[0]['episodic_return']
            if done or timesteps > 150:
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
        if config.game in ['CartPole-v1', 'Acrobot-v1']:
            n_ep = 100
        elif config.game in ['RoboschoolHalfCheetah-v1', 'RoboschoolReacher-v1']:
            n_ep = 200
        else:
            raise NotImplementedError
        perf = []
        for ep in range(n_ep):
            #print (ep)
            perf.append(self.eval_episode(self.real_env))
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
            beta_target = torch.stack([self.beta_network[i].predict(tensor(state_action_state).cuda()) for i in range(5)])
            beta_target = torch.mean(beta_target, dim=0)
        else:
            beta_target = ([self.beta_network[i].predict(torch.tensor(state_action_state).type(torch.FloatTensor)) for i in range(5)])
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

        next_actions = self.sample_action(next_states, self.std).detach()
        states_0 = tensor(config.sample_init_states())
        if self.use_cuda:
            states_0 = states_0.cuda()
        actions_0 = self.sample_action(states_0, self.std).detach()
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
