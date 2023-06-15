# Import required packages
import gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt

#@title 1.1 Choose an environment, observation mode, control mode, and reward
#@markdown Run this cell to display the action space of the chosen controller as well as the current view of the environment. The main part of the view is our view of the environment. The two views on the right are the RGB and Depth images from a third-person camera and a hand-mounted camera

# Can be any env_id from the list of Rigid-Body envs: https://haosulab.github.io/ManiSkill2/concepts/environments.html#rigid-body
# and Soft-Body envs: https://haosulab.github.io/ManiSkill2/concepts/environments.html#soft-body

# This tutorial allows you to play with 4 environments out of a total of 20 environments that ManiSkill provides
env_id = "PegInsertionSide-v0" #@param ['PickCube-v0', 'PegInsertionSide-v0', 'StackCube-v0', 'PlugCharger-v0']

# choose an observation type and space, see https://haosulab.github.io/ManiSkill2/concepts/observation.html for details
obs_mode = "pointcloud" #@param can be one of ['pointcloud', 'rgbd', 'state_dict', 'state']

# choose a controller type / action space, see https://haosulab.github.io/ManiSkill2/concepts/controllers.html for a full list
control_mode = "pd_joint_delta_pos" #@param can be one of ['pd_ee_delta_pose', 'pd_ee_delta_pos', 'pd_joint_delta_pos', 'arm_pd_joint_pos_vel']

reward_mode = "dense" #@param can be one of ['sparse', 'dense']

# create an environment with our configs and then reset to a clean state
env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
obs = env.reset()
print("Action Space:", env.action_space)
print("obs:",obs )
# take a look at the current state
img = env.render(mode="cameras")
plt.figure(figsize=(10,6))
plt.title("Current State")
plt.imshow(img)
plt.show()
env.close()