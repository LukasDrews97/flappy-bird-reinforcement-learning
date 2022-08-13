from typing import Any, Dict
import logging
import time
from datetime import datetime
from argparse import ArgumentParser
import os
import yaml
import torch
import optuna
import joblib
import functools
import pandas as pd
from flappy_bird_gym.flappy_bird_gym.envs import FlappyBirdEnvSimple, FlappyBirdEnvRGB
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym import Wrapper

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import matplotlib.pyplot as plt
max_mean_reward = -1

#torch.cuda.set_device(3)

def main(type, algorithm, policy, learning_rate, gamma, total_timesteps, name_prefix, eval_freq, model_path, frame_stack, train=True, verbose=1):
    if type == "simple":
        env, eval_env = create_simple_env(train=train)

    elif type == "rgb":
        env, eval_env = create_rgb_env(train=train, frame_stack=frame_stack)
    
    if train:
        start_time = time.strftime("%Y-%m-%d-%H_%M_%S")
        log_dir = f"./logs/{start_time}/"
        saved_models_dir = f"./saved_models/{start_time}/"
        verbose = 1
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(saved_models_dir, exist_ok=True)

        # autosave every 50000 steps
        checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=saved_models_dir, name_prefix=name_prefix)
        # evaluate every n steps
        eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=saved_models_dir+"best_models/", log_path=log_dir, eval_freq=eval_freq)
        # custom logging callback to log the score
        #logging_callback = LoggingCallBack()
        logger = configure(log_dir, ["stdout", "csv", "tensorboard", "json"])

    class_ = globals()[algorithm]
    model = class_(policy=policy, env=env, verbose=verbose, learning_rate=learning_rate, gamma=gamma)

    if train:
        model.set_logger(logger)
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    else:
        model = class_.load(model_path)

    #results_plotter.plot_results(dirs=[log_dir], num_timesteps=None, x_axis=results_plotter.X_TIMESTEPS, task_name="Text")
    #plt.show()

    #ret = evaluate_policy(model=model, env=env, n_eval_episodes=10, return_episode_rewards=True, deterministic=False)
    #print(ret)

    if not train:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render(mode="human")
            time.sleep(1/60)


def create_simple_env(train):
    # Create coordinate based environment
    env = FlappyBirdEnvSimple()
    env = Monitor(env)
    env = LoggingWrapper(env)
    env = DummyVecEnv([lambda: env for _ in range(1)])

    eval_env = None

    if train:
        eval_env = FlappyBirdEnvSimple()
        eval_env = Monitor(eval_env)
        eval_env = LoggingWrapper(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env for _ in range(1)])
    return env, eval_env


def create_rgb_env(train, frame_stack=None):
    env = FlappyBirdEnvRGB()
    env = Monitor(env)
    # Grayscale
    num_frame_stacked = 4 if (frame_stack == "None" or frame_stack == None) else int(frame_stack)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = LoggingWrapper(env)
    env = DummyVecEnv([lambda: env for _ in range(1)])
    # Use Frame stacking
    env = VecFrameStack(env, num_frame_stacked, channels_order="last")

    eval_env = None

    if train:
        eval_env = FlappyBirdEnvRGB()
        eval_env = Monitor(eval_env)
        # Grayscale
        eval_env = GrayScaleObservation(eval_env, keep_dim=True)
        eval_env = ResizeObservation(eval_env, (84, 84))
        eval_env = LoggingWrapper(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env for _ in range(1)])
        # Use Frame stacking
        eval_env = VecFrameStack(eval_env, num_frame_stacked, channels_order="last")
    return env, eval_env


def sample_ppo_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3]),
        'n_steps': trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096, 8192]),
        #'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512]),
        #'n_epochs': trial.suggest_categorical("n_epochs", [1, 5, 10, 20]),
        'gamma': trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99]),
        #'gae_lambda': trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
        #'vf_coef': trial.suggest_uniform("vf_coef", 0, 1),
        #'policy_kwargs': {
        #    'net_arch': {
        #        'small': [dict(pi=[64, 64], vf=[64, 64])],
        #        'medium': [dict(pi=[256, 256], vf=[256, 256])],
        #    }[trial.suggest_categorical("net_arch", ["small", "medium"])]
        #}
    }

def sample_a2c_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3]),
        'n_steps': trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096, 8.192]),
        'gamma': trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99]),
    }
    

def optimize_agent(trial: optuna.Trial, algorithm, total_timesteps = 1000000, env_type = 'rgb', policy = 'CnnPolicy', dir='.'):
    model_params = sample_a2c_hyperparameters(trial)
    if env_type == 'rgb':
        env, eval_env = create_rgb_env(train=True)
    else:
        env, eval_env = create_simple_env(train=True)
    class_ = globals()[algorithm]
    model = class_(policy, env, verbose=0, **model_params)
    model.learn(total_timesteps)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    global max_mean_reward
    if mean_reward > max_mean_reward:
        max_mean_reward = mean_reward
        model.save(f"{dir}/best_model_hypertuned")
        del model
    return mean_reward


class LoggingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scores = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.scores.append(int(info["score"]))
        if len(self.scores) % 1000 == 0 and len(self.scores) > 0:
            #print(sum(self.scores)/len(self.scores))
            if len(self.scores) >= 100000:
                self.scores = []
        return obs, reward, done, info

class HyperparameterTuningCallback:
    def __init__(self, n_trials):
        self.counter = 0
        self.n_trials = n_trials

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.counter += 1
        print(f'\nTrial {self.counter} of {self.n_trials} Trials')


'''
class LoggingCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallBack, self).__init__(verbose)
        self.scores = []

    def _on_step(self) -> bool:
        info_dict = self.locals['infos'][0]
        score = int(info_dict['score'])
        #self.scores.append(score)
        #sum_last_1000 = sum(self.scores[-1000:])
        #running_average_1000 = sum_last_1000 / 1000 if len(self.score) >= 1000 else sum_last_1000 / len(self.scores)
        #self.logger.record_mean("score_mean", score)
        #print("_on_step Score: ", score)
        #print(self.logger)
        return True

    def _on_rollout_end(self) -> None:
        info_dict = self.locals['infos'][0]
        score = int(info_dict['score'])
        self.logger.record_mean("score_mean", score)
'''




if __name__=="__main__": 
    '''
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--config", action="store", required=True)
    parser.add_argument("--model_path", action="store")
    args = parser.parse_args()
    args = vars(args)
    with open(args["config"], 'r') as file:
        data = yaml.safe_load(file)
    if args["test"] == True or args["train"] == False:
        train=False
    else:
        train=True

    try:
        frame_stack=data["frame_stack"]
    except KeyError:
        frame_stack=None
    '''


    start_time = time.strftime("%Y-%m-%d-%H_%M_%S")
    log_dir = f"./logs/{start_time}/"
    saved_models_dir = f"./saved_models/{start_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(direction="maximize")
    optuna_cb = HyperparameterTuningCallback(50)
    try:
        reward_func = functools.partial(optimize_agent, algorithm='PPO', total_timesteps = 1000000, env_type = 'rgb', policy = 'MlpPolicy', dir=saved_models_dir)
        study.optimize(reward_func, n_trials=50, n_jobs=50, show_progress_bar=True, callbacks=[optuna_cb])
        df = study.trials_dataframe()
        joblib.dump(study, f'{log_dir}study.pkl')
        df.to_csv(f'{log_dir}tuning_data.csv')
        #study = joblib.load("study.pkl")

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
        exit()

    '''
    main(type=data["type"], 
        algorithm=data["hyperparameter"]["algorithm"], 
        policy=data["hyperparameter"]["policy"], 
        learning_rate=float(data["hyperparameter"]["learning_rate"]),
        gamma=float(data["hyperparameter"]["gamma"]),
        total_timesteps=int(data["total_timesteps"]),
        name_prefix=data["checkpoints"]["prefix"],
        eval_freq=int(data["eval_freq"]),
        train=train, 
        model_path=args["model_path"],
        frame_stack=frame_stack,
        verbose=1)
    '''
