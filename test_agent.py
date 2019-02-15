import argparse
import os

import numpy as np
import torch

from ppo.envs import VecPyTorch, make_vec_envs
from ppo.utils import get_render_func, get_vec_normalize

import LunarLanderModel
import BipedalModel

from utils import resize_image_list

import cv2

# workaround to unpickle olf model files
# import sys
# sys.path.append('ppo')

def VideoClip(frame_stack):
    video_name = 'Bipedal.avi'

    height, width, ch = 400, 600, 3

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 80, (width, height))

    for ii, image in enumerate(frame_stack):
        video.write(image[0])
        if ii > 800:
            break

    video.release()

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

args.imsize = (40, 60)  # c, h, w
args.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.state_sequence = 4


env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, args.device, args.imsize,
                            allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

actor_critic , _ = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
actor_critic.to(args.device)



recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()
initial_imgs = resize_image_list(env.get_images(), args.imsize, args.device)
state = initial_imgs.repeat(1, args.state_sequence, 1, 1)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

accumulate_reward = []
frame_stack = []
GenVideoClip = True

for ii in range(10):
    obs = env.reset()
    initial_imgs = resize_image_list(env.get_images(), args.imsize, args.device)
    state = initial_imgs.repeat(1, args.state_sequence, 1, 1)
    episode_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                state, obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, render, reward, done, _ = env.step(action)
        next_state = torch.cat((state[:, 1:, :, :], render), 1)
        episode_reward += float(reward)
        frame_stack.append(env.get_images())

        masks.fill_(0.0 if done else 1.0)

        if done:
            accumulate_reward.append(episode_reward)
            print("Reward: ", episode_reward)
            break

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')


if GenVideoClip:
    VideoClip(frame_stack)

print("Average accumulate reward {} STD {}".format(np.mean(accumulate_reward), np.std(accumulate_reward)))

