from tabnanny import verbose
import time
from datetime import datetime
from argparse import ArgumentParser
import os
#from itertools import count
#import flappy_bird_gym
from flappy_bird_gym.flappy_bird_gym.envs import FlappyBirdEnvRGB
from stable_baselines3 import PPO
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



def main(train=True, verbose=0):
    if train:
        start_time = time.strftime("%Y-%m-%d-%H_%M_%S")
        log_dir = f"./logs/{start_time}/"
        saved_models_dir = f"./saved_models/{start_time}/"
        verbose = 1
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(saved_models_dir, exist_ok=True)

    env = FlappyBirdEnvRGB()
    env = Monitor(env)
    # Grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = LoggingWrapper(env)
    env = DummyVecEnv([lambda: env for _ in range(1)])
    # Use Frame stacking
    env = VecFrameStack(env, 8, channels_order="last")

    if train:
        eval_env = FlappyBirdEnvRGB()
        eval_env = Monitor(eval_env)
        # Grayscale
        eval_env = GrayScaleObservation(eval_env, keep_dim=True)
        eval_env = ResizeObservation(eval_env, (84, 84))
        eval_env = LoggingWrapper(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env for _ in range(1)])
        # Use Frame stacking
        eval_env = VecFrameStack(eval_env, 8, channels_order="last")

    if train:
        # autosave every 50000 steps
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=saved_models_dir, name_prefix='ppo_cnn_')
        # evaluate every n steps
        eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=saved_models_dir+"best_models/", log_path=log_dir, eval_freq=10000)
        # custom logging callback to log the score
        #logging_callback = LoggingCallBack()
        logger = configure(log_dir, ["stdout", "csv", "tensorboard", "json"])

    #TODO: Vergleich DQN
    model = PPO("CnnPolicy", env, verbose=verbose, learning_rate=1e-5, gamma=0.95)
    if train:
        model.set_logger(logger)
        model.learn(total_timesteps=10_000_000, callback=[checkpoint_callback, eval_callback])#, logging_callback])
    else:
        model = PPO.load("./saved_models/2022-07-15-12_37_38/ppo_cnn__1350000_steps")

    #results_plotter.plot_results(dirs=[log_dir], num_timesteps=None, x_axis=results_plotter.X_TIMESTEPS, task_name="Text")
    #plt.show()

    ret = evaluate_policy(model=model, env=env, n_eval_episodes=10, return_episode_rewards=True, deterministic=False)
    print(ret)

    if not train:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render(mode="human")
            time.sleep(1/60)



class LoggingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scores = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.scores.append(int(info["score"]))
        if len(self.scores) % 1000 == 0 and len(self.scores) > 0:
            print(sum(self.scores)/len(self.scores))
            if len(self.scores) >= 100000:
                self.scores = []
        return obs, reward, done, info


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
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    args = vars(args)
    if args["test"] == True or args["train"] == False:
        main(train=False)
    else:
        main(train=True)