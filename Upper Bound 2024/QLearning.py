import gymnasium as gym
import gym_puddle, json, time, cv2, random, warnings, sys
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import pyvirtualdisplay
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy as DQNPolicy
import matplotlib.collections as mc

warnings.filterwarnings("ignore")

JSON_FILE = '/Users/ishaanratanshi/Upper Bound/gym-puddle/gym_puddle/env_configs/pw3.json' 
NUM_EPISODES = 20000
EPSILON = 1
EPSILON_DECAY_RATE = 0.9995
MIN_EPSILON = 0.01
SEED = 100
LEARNING_RATE = 0.02
GAMMA = 0.99

class QLearningAgent:
    def __init__(self,
                env, 
                state_grid, 
                alpha=LEARNING_RATE, 
                gamma=GAMMA, 
                epsilon=EPSILON, 
                epsilon_decay_rate=EPSILON_DECAY_RATE, 
                min_epsilon=MIN_EPSILON):
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
        self.action_size = self.env.action_space.n  
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """
        Map a continuous state to its discretized representation
        """
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """
        Reset variables for a new episode
        """
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """
        Reset exploration rate used when training.
        """
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """
        Pick next action and update internal Q table (when mode != 'test')
        """
        state = self.preprocess_state(state)
        if mode == 'test':
            action = np.argmax(self.q_table[state])
        else:
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        self.last_state = state
        self.last_action = action
        return action

# some functions to help the visualization and interaction with the environment
def visualize(frames, video_name = "video.mp4"):
    """
    visualize environement
    """
    # Saves the frames as an mp4 video using cv2
    video_path = video_name
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def prepare_display():
    """
    Prepares display for online rendering of the frames in the game
    """
    _display = pyvirtualdisplay.Display(visible=False,size=(1400, 900))
    _ = _display.start()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')

def create_uniform_grid(low, high, bins=(20, 20)):
    """
    Define a uniformly-spaced grid that can be used to discretize a space.
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

def discretize(sample, grid):
    """
    Discretize a sample as per given grid.
    """
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    """
    Visualize original and discretized samples on a given 2-dimensional grid using matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)
    
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
        low = [splits[0] for splits in grid]
        high = [splits[-1] for splits in grid]

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 
    locs = np.stack([grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))]).T  # map discretized samples
    ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
    ax.plot(locs[:, 0], locs[:, 1], 's')  
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange')) 
    ax.legend(['original', 'discretized'])
    plt.show()

    
def run(agent, env, num_episodes=NUM_EPISODES, mode='test'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        frames = list()
        state = env.reset()[0]
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        while not done:
            state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)
            image = env.render()
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

        frames.append(image)                                                                                                                                                                                                    

    env.close()
    visualize(frames, "DQN.mp4")
    return scores


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
    q_agent = QLearningAgent(env, state_grid)
    scores = run(q_agent, env)
    plt.plot(scores); plt.title("Scores")
    plt.show()


main()