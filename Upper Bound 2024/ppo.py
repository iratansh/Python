import gymnasium as gym
import gym_puddle, json, time, cv2, random, warnings, sys, optuna
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import pyvirtualdisplay
import matplotlib.collections as mc
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

JSON_FILE = '/Users/ishaanratanshi/Upper Bound/gym-puddle/gym_puddle/env_configs/pw3.json'

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

def test_model(model, env, num_episodes):
    print('not stuck')
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            print(done)
        print(f"Episode finished with a total reward of {total_reward}")
        episode_rewards.append(total_reward)
    plot_rewards(episode_rewards)

def main():
    with open(JSON_FILE) as f:
        env_setup = json.load(f)

    env = gym.make(
    "PuddleWorld-v0",
    start=env_setup["start"],
    goal=env_setup["goal"],
    goal_threshold=env_setup["goal_threshold"],
    noise=env_setup["noise"],
    thrust=env_setup["thrust"],
    puddle_top_left=env_setup["puddle_top_left"],
    puddle_width=env_setup["puddle_width"],
    )

    vec_env = make_vec_env(lambda: env, n_envs=1)
    # env.reset()
    # model = PPO(
    #     policy=ActorCriticPolicy, 
    #     env=vec_env,
    #     # policy_kwargs=dict(net_arch=[64, 64]), 
    #     # gamma=0.99,
    #     # learning_rate=0.0001,
    #     # seed=100,
    #     # n_epochs=15,
    #     verbose=1
    # )

    # for i in range(1, 30):
    #     model.learn(total_timesteps=int(1e5), reset_num_timesteps=False, tb_log_name="PPO")
    # model.save('ppo_puddleWorld')
    
    model = PPO.load('ppo_puddleWorld', vec_env)
    test_model(model, vec_env, 20000)

    env.close()

main()

