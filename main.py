import time
from itertools import count
import flappy_bird_gym
#from flappy_bird_gym import flappy_bird_gym
#import flappy_bird_gym.flappy_bird_gym
from flappy_bird_gym.flappy_bird_gym.envs import FlappyBirdEnvRGB
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


def main():
    #env = flappy_bird_gym.flappy_bird_gym.make("FlappyBird-rgb-v0") #FlappyBird-rgb-v0
    env = FlappyBirdEnvRGB()
    #check_env(env, skip_render_check=False)
    #env = make_vec_env("FlappyBird-rgb-v0", n_envs=2)
    
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_test")
    #model = PPO.load("ppo_test")

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render(mode="human")
        #time.sleep(1/30)
        if dones:
            obs = env.reset()


    '''
    obs = env.reset()

    for _ in count():
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        #env.render()
        #time.sleep(1/15)

        if done:
            env.reset()
    env.close()
    '''















if __name__=="__main__": 
    main()