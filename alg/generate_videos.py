import numpy as np
import gym
import panda_gym
import cv2
import os
import sys
from time import sleep

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper

SAVE_DIR = os.path.join(PARENT_DIR, "demo_data", "PickAndPlace")
os.makedirs(SAVE_DIR, exist_ok=True)


def save_state_trajectory(state_traj, index):
    traj_path = os.path.join(SAVE_DIR, f"state_traj_{index}.csv")
    np.savetxt(traj_path, state_traj, delimiter=' ')
    print(f"✅ Saved trajectory to {traj_path}")


def record_video(env, action_sequence, filename, init=None):
    video_path = os.path.join(SAVE_DIR, f"{filename}.mp4")

    height = 480
    width = 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    step = 0
    done = False

    if init is not None:
        obs = env.reset(whether_random=False, object_pos=init)
    else:
        obs = env.reset(whether_random=True)

    while not done and step < len(action_sequence):
        action = action_sequence[step]
        obs, reward, done, info = env.step(action)

        img_tuple = env.render(mode='rgb_array')
        img = np.array(img_tuple, dtype=np.uint8)
        img = img.reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)
        step += 1

    out.release()
    print(f"🎥 Saved video to {video_path}")


def prepare_demo_pool(demo_path):
    state_traj = np.genfromtxt(demo_path + 'state_traj.csv', delimiter=' ')
    action_traj = np.genfromtxt(demo_path + 'action_traj.csv', delimiter=' ')
    next_state_traj = np.genfromtxt(demo_path + 'next_state_traj.csv', delimiter=' ')
    reward_traj = np.genfromtxt(demo_path + 'reward_traj.csv', delimiter=' ')
    done_traj = np.genfromtxt(demo_path + 'done_traj.csv', delimiter=' ')

    reward_traj = np.reshape(reward_traj, (-1, 1))
    done_traj = np.reshape(done_traj, (-1, 1))

    starting_ids = [i for i in range(state_traj.shape[0]) if state_traj[i][0] == np.inf]
    total_demo_num = len(starting_ids)

    demos = []
    for i in range(total_demo_num):
        start_step_id = starting_ids[i]
        end_step_id = starting_ids[i + 1] if i < total_demo_num - 1 else state_traj.shape[0]

        demo = {
            'state_trajectory': state_traj[(start_step_id + 1):end_step_id, :],
            'action_trajectory': action_traj[(start_step_id + 1):end_step_id, :],
            'next_state_trajectory': next_state_traj[(start_step_id + 1):end_step_id, :],
            'reward_trajectory': reward_traj[(start_step_id + 1):end_step_id, :],
            'done_trajectory': done_traj[(start_step_id + 1):end_step_id, :]
        }
        demos.append(demo)

    return demos


def generate_expert_videos():
    demo_path = os.path.join(PARENT_DIR, "demo_data", "PickAndPlace/")
    demos = prepare_demo_pool(demo_path)

    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    for i, demo in enumerate(demos[:20]):
        index = i
        init_pos = demo['state_trajectory'][0][7:10]
        record_video(env, demo['action_trajectory'], f"clip_{index}", init=init_pos)
        save_state_trajectory(demo['state_trajectory'], index)


def generate_random_videos():
    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    for i in range(10):
        obs = env.reset(whether_random=True)
        done = False
        states = []
        actions = []
        step = 0

        while not done and step < 150:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            state = np.concatenate([obs["observation"], obs["achieved_goal"]])
            states.append(state)
            actions.append(action)
            step += 1

        index = i + 20  # random demos start at index 20
        record_video(env, actions, f"clip_{index}")
        save_state_trajectory(np.array(states), index)


if __name__ == '__main__':
    generate_expert_videos()
    generate_random_videos()
