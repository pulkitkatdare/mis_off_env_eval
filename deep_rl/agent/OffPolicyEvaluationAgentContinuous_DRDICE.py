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




class OffPolicyEvaluationContinuous_DRDICE(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.DICENet = config.dice_net_fn()
        self.DICENet_target = config.dice_net_fn()
        self.DICENet_target.load_state_dict(self.DICENet.state_dict())
        self.network = config.network_fn()
        self.replay = config.replay_fn()
        self.total_steps = 0
        self.data_collection_noise = config.data_collection_noise
        self.with_beta = config.with_beta

        try:
            self.replay.load('./data/GradientDICE/%s-data%d' % (config.game, config.dataset))
        except:
            pass
        
        #self.load('./data/GradientDICE/%s-policy' % config.game)
        self.expert_policy = config.expert_policy
        self.env_p = config.environment_p
        self.env_q = config.environment_q
        self.deterministic = config.deterministic
        self.noise_std = config.noise_std
        self.data_collection_noise = config.data_collection_noise
        self.beta_network = config.beta_factor
        self.loss_history = []
        self.min_loss = 1e6
        
        #self.oracle_perf = self.load_oracle_perf()
        #print('True performance: %s' % (self.oracle_perf))

    def collect_data(self):
        self.sample_trajectory(self.data_collection_noise)
        self.replay.save('./data/GradientDICE/%s-data%d' % (self.config.game, self.config.dataset))

    def sample_trajectory(self, std):
        config = self.config
        for i in range(100):
            print('Sampling trajectory %s' % (i))
            states = self.env_q.reset()
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            for j in range(150):
                action, lstm_states = self.expert_policy.predict(
                states,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=self.deterministic)
                next_states, rewards, done, info = self.env_q.step(action)
                action += np.random.normal(size=(6, ))*1e-4
                action += np.random.normal(size=(6,))*std
                #if np.random.rand() < std:
                #    action = np.random.randint(3)
                #ret = info[0]['episodic_return']
                if done or j == 149:
                    print('Episode end')
                    break
                input_states = np.reshape(states, (1, 26))
                input_rewards = np.zeros((1, 1))
                input_rewards[0, 0] = rewards
                input_actions = np.reshape(action, (1, 6))#
                #input_actions[0, :] = action
                input_next_states = np.reshape(next_states, (1, 26))
                input_done = np.zeros((1, 1))
                input_done[0, 0] = done
                
                experiences = list(zip(input_states, input_actions, input_rewards, input_next_states, input_done))
                self.replay.feed_batch(experiences)
                states = next_states

    def sample_action(self, states, std):
        lstm_states=None
        episode_start = np.ones((1,), dtype=bool)
        actions, lstm_states = self.expert_policy.predict(
                states,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=self.deterministic)
        #print (np.shape(actions))
        #actions = np.array(actions)
        actions = torch.from_numpy(actions).type(torch.FloatTensor)
        #print (actions.size())
        actions += torch.randn(actions.size()) * std
        #randomness = torch.rand(actions.size())
        #whether_rand_action =  (randomness<std)
        #actions = (~whether_rand_action)*actions + whether_rand_action*torch.randint(low=0, high=3, size=actions.size())
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
            #action = np.reshape(action, (6,))
            #print (action)
            action += np.random.normal(size=(6, ))*1e-4
            #print (np.shape(action))
            action += np.random.normal(size=(6, ))*self.noise_std 
            
            state, reward, done, info = environment.step(action)
            rewards.append(reward)
            ret = None#info[0]['episodic_return']
            if done or timesteps > 150:
                #print('Computing true performance: %s' % ret)
                break
        print (timesteps)
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
        else:
            raise NotImplementedError
        perf = []
        for ep in range(n_ep):
            print (ep)
            perf.append(self.eval_episode(self.env_p))
        if config.discount == 1:
            return np.mean(perf)
        else:
            return (1 - config.discount) * np.mean(perf)

    def step(self):
        config = self.config
        if config.correction == 'no':
            return
        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        state_action_state = np.concatenate((states, actions, next_states), axis=1)
        beta_target = self.beta_network.predict(torch.tensor(state_action_state).type(torch.FloatTensor))
        #beta_target = beta_target/beta_target.sum()
        #print (beta_target)
        
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        next_states = tensor(next_states)
        masks = tensor(1 - terminals).unsqueeze(-1)

        next_actions = self.sample_action(next_states, config.noise_std).detach()
        states_0 = tensor(config.sample_init_states())
        actions_0 = self.sample_action(states_0, config.noise_std).detach()

        tau = self.DICENet.tau(states, actions)
        f = self.DICENet.f(states, actions)
        #print (np.shape(next_states))
        #print (np.shape(next_actions))
        f_next = self.DICENet.f(next_states, next_actions)
        f_0 = self.DICENet.f(states_0, actions_0)
        u = self.DICENet.u(states.size(0))

        tau_target = self.DICENet_target.tau(states, actions).detach()
        f_target = self.DICENet_target.f(states, actions).detach()
        f_next_target = self.DICENet_target.f(next_states, next_actions).detach()
        f_0_target = self.DICENet_target.f(states_0, actions_0).detach()
        u_target = self.DICENet_target.u(states.size(0)).detach()

        if config.correction == 'GenDICE':
            if self.with_beta:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next - \
                        tau_target * (f + 0.25 * f.pow(2)) + config.lam * (u * tau_target - u - 0.5 * u.pow(2)))*beta_target
                J_convex = (1 - config.discount) * f_0_target + (config.discount * tau * f_next_target - \
                       tau * (f_target + 0.25 * f_target.pow(2)) + \
                       config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2)))*beta_target
            else:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next - \
                    tau_target * (f + 0.25 * f.pow(2)) + config.lam * (u * tau_target - u - 0.5 * u.pow(2)))
                J_convex = (1 - config.discount) * f_0_target + config.discount * tau * f_next_target - \
                    tau * (f_target + 0.25 * f_target.pow(2)) + \
                       config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2))


        elif config.correction == 'GradientDICE':
            if self.with_beta:
                J_concave = (1 - config.discount) * f_0 + (config.discount * tau_target * f_next - \
                            tau_target * f - 0.5 * f.pow(2) + config.lam * (u * tau_target - u - 0.5 * u.pow(2)))*beta_target
                J_convex = (1 - config.discount) * f_0_target + (config.discount * tau * f_next_target - \
                        tau * f_target - 0.5 * f_target.pow(2) + \
                        config.lam * (u_target * tau - u_target - 0.5 * u_target.pow(2)))*beta_target
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
        self.DICENet.opt.step()
        if self.total_steps % config.target_network_update_freq == 0:
            self.DICENet_target.load_state_dict(self.DICENet.state_dict())

        self.total_steps += 1

    def eval_episodes(self):
        experiences = self.replay.sample(1000)#len(self.replay.data))
        states, actions, rewards, next_states, terminals = experiences
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        if self.config.correction == 'no':
            tau = 1
        else:
            tau = self.DICENet.tau(states, actions)
        perf = (tau * rewards).mean()
        loss = (perf - self.oracle_perf).pow(2).mul(0.5)
        self.loss_history.append(loss)
        if loss.item() < self.min_loss:
            file_appender = '150_100' + str(int(10*self.noise_std)) + '_' + str(int(10*self.data_collection_noise)) + '_' + str(150)
            filename = './log_dr_halfcheetah' +'/' + self.config.correction + '_' + file_appender + '_' + str(self.config.index) + '.ptr'
            torch.save(self.DICENet.state_dict(), filename)
        print('perf_loss: %s' % (loss))
        self.logger.add_scalar('perf_loss', loss)
