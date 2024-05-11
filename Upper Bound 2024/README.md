References: https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb, https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html, https://www.youtube.com/watch?v=EUrWGTCGzlA&ab_channel=JohnnyCode

This program implements a Q-learning agent to navigate a reinforcement learning environment provided by the "PuddleWorld-v0" environment from the gymnasium library. 

1. **Q-Learning Agent**: The core of the program is the QLearningAgent class, which implements the Q-learning algorithm. This algorithm learns to make decisions by iteratively updating a Q-table based on observed rewards and actions.

2. **Technologies Used**:
   - **Python**: The programming language used for the entire program.
   - **gymnasium**: A toolkit for developing and comparing reinforcement learning algorithms.
   - **Stable Baselines3**: A set of high-quality implementations of reinforcement learning algorithms in Python.
   - **matplotlib**: A plotting library used for visualizing scores during training.
   - **numpy**: A library for numerical computing used extensively for array operations.
   - **OpenCV (cv2)**: A library mainly aimed at real-time computer vision, used for video processing in this case.
   - **pyvirtualdisplay**: A library used for creating a virtual display required for rendering frames in the environment.
   - **json**: A standard Python library for parsing JSON files.

3. **Functionality**:
   - **Visualization**: The program provides functions for visualizing the environment and samples during training.
   - **Grid Creation**: It defines a uniformly-spaced grid to discretize the continuous state space of the environment.
   - **Training and Testing**: The run() function executes the agent in the environment, allowing it to learn and make decisions. It can run in either 'train' or 'test' mode.
   - **Main Function**: The main() function initializes the environment, creates a Q-learning agent, runs the training, and visualizes the results using matplotlib.

4. **Environment Setup**:
   - The environment setup, including parameters like start position, goal, noise, thrust, and puddle configuration, is loaded from a JSON file.

Overall, this program demonstrates how to implement a Q-learning agent to solve a reinforcement learning problem using the gymnasium library and associated tools.


