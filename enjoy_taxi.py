import gym
import numpy as np
import random
from os import system, name
from time import sleep
import pickle as pkl 
from utils.taxi_environment import taxi_with_transitions
import argparse

sim_p = 0.0
real_p = 0.1

def decode(state, env):
    decode_state = list(env.decode(state))
    decode_state = np.array(decode_state).reshape(1, -1)
    return decode_state


def collect_data(display_episodes, policy_params, transition_probability, filename):
    total_epochs, total_penalties, total_rewards = 0, 0, 0
    env = taxi_with_transitions(transition_probability=transition_probability)
    q_table = pkl.load(open('./logs/dqn/taxi_v3_1.pkl', 'rb'))
    data = []

    for episode in range(display_episodes):
        state = env.reset()
        epochs, penalties, rewards = 0, 0, 0
        
        done = False
        
        for _ in range(150):
            if np.random.uniform() < policy_params:
                action = np.random.randint(6)
            else:
                action = np.argmax(q_table[state])
            #action = np.argmax(q_table[state])
            next_state, reward, done, info = env.step(action)
            rewards += (0.99**epochs)*reward
            if reward == -10:
                penalties += 1

            epochs += 1
            #sleep(0.15) # Sleep so the user can see the

            data.append((decode(state, env.env), action, decode(next_state, env.env), reward, done))
            state = next_state
        if (episode%100) == 0:
            print (episode, epochs)
        total_penalties += penalties
        total_epochs += epochs
        total_rewards += rewards
    print(f"Results after {display_episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / display_episodes}")
    print(f"Average penalties per episode: {total_penalties / display_episodes}")
    print(f"Average rewards per episode: {total_rewards / display_episodes}")


    with open('./offline_data/' + filename + '.pkl', 'wb') as f:
        pkl.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_noise", type=float, default=0.1
    )
    parser.add_argument(
        "--save_file", type=str, default="transitions"
    )
    parser.add_argument(
        "--transition_probability", type=float, default=1
    )
    args = parser.parse_args()


    collect_data(display_episodes=100, policy_params=args.policy_noise, transition_probability=args.transition_probability, filename=args.save_file)