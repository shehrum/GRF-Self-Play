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




"""Script for multi-agent wrapper for Google Research Football environment compatible with RLlib. The base code
   was provided with Google Research Football examples in the run_multiagent_rllib.py file which was modified for the project
"""


from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
import gfootball.env as football_env
import numpy as np

class RllibGFootball(MultiAgentEnv):
  """Wrapper for GFootball to make it compatible with rllib."""

  def __init__(self, config):
    self.mode = config["mode"]
    if self.mode == 'test':
        print("\n++++++++++++++++++++RUNNING EVALUATION MODE++++++++++++++++++++++\n")
        # Testing mode
        self.env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic', stacked=True,
        logdir='/tmp/rllib_test',
        write_goal_dumps=False, write_full_episode_dumps=False, render=False,
        dump_frequency=0,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,rewards='scoring,checkpoints')

        self.num_agents = 1
    else:
        # Training mode
        self.env = football_env.create_environment(
          env_name='11_vs_11_easy_stochastic', stacked=True,
          logdir='/tmp/rllib_test',
          write_goal_dumps=False, write_full_episode_dumps=False, render=False,
          dump_frequency=0,
          number_of_left_players_agent_controls=1,
          number_of_right_players_agent_controls=1,rewards='scoring,checkpoints')

        self.num_agents = 2
    self.observation_space = gym.spaces.Box(0,255,[72,96,16],dtype=np.float32)
    self.action_space = gym.spaces.Discrete(19)
    
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
  def step(self, action_dict):
    actions = []
    for key, value in sorted(action_dict.items()):
      actions.append(value)
    o, r, d, i = self.env.step(actions)
    rewards = {}
    obs = {}
    infos = {}
    for pos, key in enumerate(sorted(action_dict.keys())):
      infos[key] = i
      if self.num_agents > 1:
        rewards[key] = r[pos]
        obs[key] = o[pos]
      else:
        rewards[key] = r
        obs[key] = o
    dones = {'__all__': d}
    return obs, rewards, dones, infos

