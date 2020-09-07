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

"""Script for evaluating/rendering training agent games in Google Research Football against easy computer bot. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

import numpy as np
import argparse
import gfootball.env as football_env
import gym
import ray

from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo import PPOTrainer

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.utils.annotations import override
_, nn = try_import_torch()
import torch.nn.functional as F
import torch
import json
import pdb
from eval_environment import RllibGFootball
import gfootball.env as football_env
from model import CustomTorchModel

parser = argparse.ArgumentParser()
parser.add_argument('--num-agents', type=int, default=1)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')


# Policy mapper for evaluation
def policy_mapping_fn_eval(agent_id):
    return "policy_01"

# Train policy1 all the time for agent01, and randomly sample from other policies for opponent
def policy_mapping_fn(agent_id):
    if agent_id.startswith("agent_01"):
        return "policy_01" # Choose 01 policy for agent_01
    else:
      # Define probability split for sampling policies
        return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"],1,
                                p=[.8, .2/3, .2/3, .2/3])[0]


if __name__ == '__main__':
    args = parser.parse_args()
    # Initializing ray
    ray.init()
    # Registering environment
    ModelCatalog.register_custom_model("my_model", CustomTorchModel)
    register_env('gfootball', lambda _: RllibGFootball(args.num_agents))

    # Creating instance of environment
    single_env = RllibGFootball(args.num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space


#   PPO trainer
    ppo_trainer = PPOTrainer(
      env="gfootball",
      config={
        "framework": "torch",
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    'simple_optimizer': args.simple,
    'observation_filter': 'NoFilter',
          "num_envs_per_worker": 1,
          "num_gpus":1,
          "ignore_worker_failures": True,
          "train_batch_size": 4000,
          'rollout_fragment_length':512,
          "sgd_minibatch_size": 500,
          "lr": 3e-4,
          "lambda": .95,
          "gamma": .998,
          "entropy_coeff": 1e-4,
          "kl_coeff": 1.0,
          "clip_param": 0.2,
          "num_sgd_iter": 10,
          "vf_share_layers": True, #?? True?
          "vf_clip_param": 100.0,
          "model": {
            "custom_model": "my_model"

          },
          "multiagent": {
              "policies": {
                  "policy_01": (None, obs_space, act_space, {}),
                  "policy_02": (None, obs_space, act_space, {}),
                  "policy_03": (None, obs_space, act_space, {}),
                  "policy_04": (None, obs_space, act_space, {})

              },
              "policy_mapping_fn": tune.function(policy_mapping_fn),
              "policies_to_train": ["policy_01"]
          },

        "evaluation_interval": 50,
        "evaluation_config": {
          "env_config": {
            # Use test set to evaluate
            'mode': 'test'
          },
            "explore": False,
            "multiagent": {
                "policies": {
                    # the first tuple value is None -> uses default policy class
                    "policy_01": (None, obs_space, act_space, {}),
                    "policy_02": (None, obs_space, act_space, {}),
                    "policy_03": (None, obs_space, act_space, {}),
                    "policy_04": (None, obs_space, act_space, {})

                },
                "policy_mapping_fn": policy_mapping_fn_eval
            },
        },

      })
# Uncomment to Load check-point, replace the file path with checkpoint to use
    # checkpoint_file = '/Users/shehrum/Documents/ray_results/PPO_gfootball_2020-08-26_11-19-1572zo9cii/checkpoint_2601/checkpoint-2601'
    # ppo_trainer.restore(checkpoint_file)


# Defining action dictionary for histogram
    action_dict = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,
       '10':0,'11':0,'12':0,'13':0,'14':0,'15':0,'16':0,'17':0,'18':0}
    action_names =  {'0':'idle','1':'left','2':'top_left','3':'top','4':'top_right',
                 '5':'right','6':'bottom_right','7':'bottom','8':'bottom_left','9':'long_pass',
       '10':'high_pass','11':'short_pass','12':'shot','13':'sprint','14':'release_direction',
                 '15':'release_sprint','16':'sliding','17':'dribble','18':'release_dribble'}

    # Create environment object
    # To render games, change set write_goal_dumps=True, write_full_episode_dumps=True,write_video=True ,render=True
    env = football_env.create_environment(env_name="11_vs_11_easy_stochastic", stacked=True, logdir='logs', write_goal_dumps=False, write_full_episode_dumps=False,write_video=False ,render=False,number_of_left_players_agent_controls=1 ,number_of_right_players_agent_controls=0,rewards='scoring,checkpoints')
    env.reset()
    steps = 0
    draw = 0
    wins = 0
    losses=0
    # Select no. of episodes
    n_episodes=100
    avg_rew=0
    total_rew=0


    #  Play game with trained agent
    for i_episode in range(1, n_episodes+1):
        obs=env.reset()
        episode_rew=0
        while True:
            # Sample action from trained policy
            action=ppo_trainer.compute_action(obs,policy_id='policy_01')
            action_dict[str(int(action))]+=1
            obs, rew, done, info = env.step(action)
            episode_rew+=rew

            if done:
                break

        if episode_rew>=0 and episode_rew<=1.0:
            draw+=1
        if episode_rew>1.0:
            wins+=1
        total_rew+=episode_rew
        
        print("Episode:%d Reward:%.2f" % (i_episode, episode_rew))
    avg_rew = total_rew / i_episode
    print("Average over %d episodes: %.2f" % (i_episode, avg_rew))
    losses=100-(draw+wins)
    print('total draws: %d wins: %d losses: %d' %(draw,wins,losses))

#  Save information for plotting histogram of action

    plot_dict={}
    for i in range(19):
        plot_dict[action_names[str(i)]]=action_dict[str(i)]

    with open('plot_dict_lr.pickle', 'wb') as handle:
        pickle.dump(plot_dict, handle)




# Un-comment for plotting histogram
    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)


    
    # plt.bar(plot_dict.keys(),height=plot_dict.values()) 
    # plt.xticks(rotation=90)
    # plt.ylabel('Frequency')
    # plt.xlabel('Action')
    # fig = plt.figure(figsize=(11, 11))
    # plt.savefig('action_distribution_plot2.png',bbox_inches='tight')


