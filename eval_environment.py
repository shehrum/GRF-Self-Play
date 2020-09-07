# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




"""Script for multi-agent wrapper for Google Research Football eval environment compatible with RLlib. 
   The base code was provided with Google Research Football examples in the run_multiagent_rllib.py file which was
    modified for the project
"""


from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
import gfootball.env as football_env
import numpy as np


# Since we use a different environment setting for evaluation, we modify the functions to be compatible with single agent

class RllibGFootball(MultiAgentEnv):
  """A wrapper for GFootball to make it compatible with rllib."""

  def __init__(self, num_agents):
    self.env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic', stacked=True,
        logdir='logs',
        write_goal_dumps=False, write_full_episode_dumps=False, render=False,
        dump_frequency=0,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,rewards='scoring,checkpoints')
 
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    self.num_agents = 1

# Defining environment reset to be compatible with RLlib
  def reset(self):
    original_obs = self.env.reset()
    obs = {}
    for x in range(self.num_agents):
      if self.num_agents > 1:
        obs['agent_%d' % x] = original_obs[x]
      else:
        obs['agent_%d' % x] = original_obs
    return obs

# Step function to perform a single action on the environment
  def step(self, action):

    obs, rewards, dones, infos = self.env.step(action)
    return obs, rewards, dones, infos


 
def policy_mapping_fn_eval(agent_id):
    return "policy_01"
# Train policy1 all the time for agent01, and randomly sample from other policies for opponent
def policy_mapping_fn(agent_id):
    if agent_id.startswith("agent_01"):
        return "policy_01" # Choose 01 policy for agent_01
    else:
        return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"],1,
                                p=[.8, .2/3, .2/3, .2/3])[0]
