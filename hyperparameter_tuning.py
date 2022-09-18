from typing import Any, Dict
import time
from datetime import datetime
from argparse import ArgumentParser
import os
import torch
import optuna
import joblib
import functools
import pandas as pd
from flappy_bird_gym.flappy_bird_gym.envs import FlappyBirdEnvSimple, FlappyBirdEnvRGB
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from train_agent import create_simple_env, create_rgb_env, LoggingWrapper

#torch.cuda.set_device(3)

max_mean_reward = -1


def sample_ppo_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    '''
    Specifies the possible hyperparameters for the PPO algorithm.
    '''
    return {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3]),
        'n_steps': trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096, 8192]),
        'gamma': trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99]),
    }

def sample_a2c_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    '''
    Specifies the tunable hyperparameters for the A2C algorithm.
    '''
    return {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3]),
        'n_steps': trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096, 8192]),
        'gamma': trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99]),
    }
    

def optimize_agent(trial: optuna.Trial, algorithm, total_timesteps = 200000, env_type = 'rgb', policy = 'CnnPolicy', dir='.', sample_func=sample_ppo_hyperparameters):
    '''
    Function to be maximized. The best model is saved in the provided directory.
    Args:
        trial: 
            Object providing interfaces for parameter suggestion.
        algorithm:
            The algorithm to be used. Can be either "PPO" or "A2C".
        total_timesteps:
            The number of timesteps this trial gets executed.
        env_type:
            The type of environment to be used. Can be either "simple" or "rgb".
        policy:
            The type of policy to be used. Can be either "CnnPolicy" or "MlpPolicy".
        dir:
            The path to the output directory. 
        sample_func:
            The function to sample hyperparameters from.
    Returns:
        Reward achieved with the sampled parameters.
    '''
    # Get params
    model_params = sample_func(trial)

    # Create environment
    if env_type == 'rgb':
        env, eval_env = create_rgb_env(train=True)
    else:
        env, eval_env = create_simple_env(train=True)

    # Instantiate agent and learn
    class_ = globals()[algorithm]
    model = class_(policy, env, verbose=0, **model_params)
    model.learn(total_timesteps)

    # evaluate agent
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    # save model if it's the best one yet
    global max_mean_reward
    if mean_reward > max_mean_reward:
        max_mean_reward = mean_reward
        model.save(f"{dir}/best_model_hypertuned")
        del model
    return mean_reward


class HyperparameterTuningCallback:
    '''
    A wrapper for the tuning process. Monitors and prints the progress.
    '''
    def __init__(self, n_trials):
        self.counter = 0
        self.n_trials = n_trials

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.counter += 1
        print(f'\nTrial {self.counter} of {self.n_trials} Trials')


if __name__=="__main__": 
    # Parse CLI parameters
    parser = ArgumentParser()
    parser.add_argument("--algorithm", action="store", required=True)
    parser.add_argument("--timesteps", action="store", required=True)
    parser.add_argument("--env", action="store", required=True)
    parser.add_argument("--policy", action="store", required=True)
    parser.add_argument("--trials", action="store", required=True)
    args = parser.parse_args()
    args = vars(args)

    try:
        timesteps = int(args["timesteps"])
    except ValueError:
        print(f"{timesteps} is not a valid number")
        exit()

    try:
        trials = int(args["trials"])
    except ValueError:
        print(f"{trials} is not a valid number")
        exit()

    env = args["env"].lower()
    if env not in ["simple", "rgb"]:
        print("Env must be one of [simple, rgb]")
        exit()        

    policy = args["policy"].lower()
    if policy not in ["cnn", "mlp"]:
        print("Policy must be one of [CNN, MLP]")
        exit()        

    algorithm = args["algorithm"].lower()
    if algorithm not in ["ppo", "a2c"]:
        print("Algorithm must be one of [PPO, A2C]")
        exit()

    policy = f"{policy[:1].upper()}{policy[1:]}Policy"
    sample_func = sample_ppo_hyperparameters if algorithm == "ppo" else sample_a2c_hyperparameters

    # Create directories to save models and logs
    start_time = time.strftime("%Y-%m-%d-%H_%M_%S")
    log_dir = f"./logs/{start_time}/"
    saved_models_dir = f"./saved_models/{start_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create study
    study = optuna.create_study(direction="maximize")
    optuna_cb = HyperparameterTuningCallback(trials)
    try:
        reward_func = functools.partial(optimize_agent, algorithm=algorithm.upper(), total_timesteps = timesteps, env_type = env, policy = policy, dir=saved_models_dir, sample_func=sample_func)
        study.optimize(reward_func, n_trials=trials, n_jobs=-1, show_progress_bar=False, callbacks=[optuna_cb])

        # Save study and parameters
        df = study.trials_dataframe()
        joblib.dump(study, f'{log_dir}study.pkl')
        df.to_csv(f'{log_dir}tuning_data.csv')

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
        exit()
