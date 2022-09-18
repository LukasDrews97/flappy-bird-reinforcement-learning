from typing import Any, Dict
import time
from datetime import datetime
from argparse import ArgumentParser
import os
import yaml
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym import Wrapper

from flappy_bird_gym.flappy_bird_gym.envs import FlappyBirdEnvSimple, FlappyBirdEnvRGB

#torch.cuda.set_device(3)


def main(type, algorithm, policy, learning_rate, gamma, total_timesteps, name_prefix, eval_freq, model_path, frame_stack, train=True, verbose=1):
    '''
    Trains or tests a given model.
    Args:
        env_type:
            The type of environment to be used. Can be either "simple" or "rgb".
        algorithm:
            The algorithm to be used. Can be either "PPO" or "A2C".
        policy:
            The type of policy to be used. Can be either "CnnPolicy" or "MlpPolicy".
        learning_rate:
            The learning rate to be used.
        gamma:
            The discount factor to be used.
        total_timesteps:
            The number of timesteps the agents learns for.
        name_prefix:
            Prefix for output files.
        eval_freq:
            The number of timesteps the agent gets evaluated after.
        model_path:
            The path to the saved model if in testing mode.
        frame_stack:
            The number of frames to be stacked if in rgb environment.
        train:
            Determines if a model is trained or a trained model gets executed.
        verbose:
            Set debugging verbosity level. 
    '''
    # Create environment
    if type == "simple":
        env, eval_env = create_simple_env(train=train)

    elif type == "rgb":
        env, eval_env = create_rgb_env(train=train, frame_stack=frame_stack)
    
    # Train a model
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
        logger = configure(log_dir, ["stdout", "csv", "json"])

    class_ = globals()[algorithm]
    model = class_(policy=policy, env=env, verbose=verbose, learning_rate=learning_rate, gamma=gamma)

    if train:
        model.set_logger(logger)
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    else:
        model = class_.load(model_path)

    # Run a model
    if not train:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render(mode="human")
            time.sleep(1/60)


def create_simple_env(train):
    '''
    Create the simple environment. If train is True, create a seperate environment for testing.
    '''
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
    '''
    Create the rbg environment. If train is True, create a seperate environment for testing.
    '''
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


class LoggingWrapper(Wrapper):
    '''
    A wrapper for the learning process. Monitors and prints the progress.
    '''
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



if __name__=="__main__": 
    # Parse CLI parameters
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

    # Create directories to save models and logs
    start_time = time.strftime("%Y-%m-%d-%H_%M_%S")
    log_dir = f"./logs/{start_time}/"
    saved_models_dir = f"./saved_models/{start_time}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

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
