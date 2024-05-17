"""
Optuna Hyperparameter tuning for Puddleworld PPO model
"""

import os, time, random, json, pprint, optuna, gym_puddle, numpy as np
import matplotlib.pyplot as plt
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import pickle as pkl
from stable_baselines3.common.vec_env import DummyVecEnv

study_path = "/Users/ishaanratanshi/Upper Bound/Tests"
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

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""
    def __init__(self, eval_env, trial, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))

def sample_ppo_params(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 0.1, log=True),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'n_epochs': trial.suggest_int('n_epochs', 1, 48),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0)
    }


def create_uniform_grid(low, high, bins=(60, 60)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

def create_discretized_env():
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
    return env

def objective(trial: optuna.Trial) -> float:
    time.sleep(random.random() * 16)
    env = create_discretized_env()
    env = make_vec_env(lambda: env)

    sampled_hyperparams = sample_ppo_params(trial)

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    model = PPO("MlpPolicy", env=env, seed=None, verbose=0, tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=5, eval_freq=10000, deterministic=False, callback_after_eval=stop_callback
    )

    params = sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(10000000, callback=eval_callback)
    except (AssertionError, ValueError) as e:
        print(e)
        print("============")
        print("Sampled params:")
        pprint.pprint(params)
        raise optuna.exceptions.TrialPruned()
    finally:
        env.close()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward

if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=4, n_trials=128)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)
