"""
PPO model for Puddleworld environment
"""

import gymnasium as gym
import gym_puddle, json, time, cv2, random, warnings, sys, optuna, pyvirtualdisplay
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mc
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

JSON_FILE = '/Users/ishaanratanshi/Upper Bound/gym-puddle/gym_puddle/env_configs/pw3.json'

class DiscretizedEnv(gym.Wrapper):
    def __init__(self, env, state_grid):
        super().__init__(env)
        self.state_grid = state_grid

    def reset(self, **kwargs):
        obs_tuple = self.env.reset(**kwargs)
        obs = obs_tuple 
        return discretize(obs[0], self.state_grid)

    def step(self, action):
        observation, reward, done, trunc, info  = self.env.step(action)
        return discretize(observation, self.state_grid), reward, done, trunc, info


def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

def test_model(model, env, num_episodes):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Episode finished with a total reward of {total_reward}")
        episode_rewards.append(total_reward)
    plot_rewards(episode_rewards)

def visualize(frames, video_name = "video.mp4"):
    # Saves the frames as an mp4 video using cv2
    video_path = video_name
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def prepare_display():
    # Prepares display for online rendering of the frames in the game0
    _display = pyvirtualdisplay.Display(visible=False,size=(1400, 900))
    _ = _display.start()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')

def create_uniform_grid(low, high, bins=(60, 60)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

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

    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high)
    env = DiscretizedEnv(env, state_grid)

    vec_env = make_vec_env(lambda: env, n_envs=1)
    env.reset()
    model = PPO(
        policy=ActorCriticPolicy, 
        env=vec_env,
        policy_kwargs=dict(net_arch=[64, 64]), 
        gamma=0.99,
        learning_rate=0.1,
        seed=100,
        n_epochs=15,
        verbose=1
    )
    # model.learn(total_timesteps=int(1e5), reset_num_timesteps=False, tb_log_name="PPO")
    # model.save('ppo_puddleWorld')

    # for i in range(1, 30):
    #     model.learn(total_timesteps=int(1e5), reset_num_timesteps=False, tb_log_name="PPO")
    # model.save('ppo_puddleWorld')
    
    # model = PPO.load('ppo_puddleWorld', vec_env)
    # test_model(model, vec_env, 20000)

    env.close()

main()

