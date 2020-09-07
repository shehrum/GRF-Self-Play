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

"""Script for training agent in Google Research Football with competitive self-play. The base code was provided with
   Google Research Football examples which was heavily modified for the project
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import gfootball.env as football_env
import gym
import ray

from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.utils import merge_dicts

from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.utils.annotations import override

from model import CustomTorchModel
_, nn = try_import_torch()
import torch.nn.functional as F
import torch
import pdb
from environment import RllibGFootball



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
  ray.init(num_gpus=1)

  # Simple environment with `num_agents` independent players
  ModelCatalog.register_custom_model("my_model", CustomTorchModel)
  # ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

  register_env('gfootball', lambda _: RllibGFootball({"mode":"train"}))

  single_env = RllibGFootball({"mode":"train"})
  obs_space = single_env.observation_space
  act_space = single_env.action_space

  # Modify apex config
  config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
  config["target_network_update_freq"] = 50000
  config["num_workers"] = 4
  config["num_envs_per_worker"] = 4
  config["gamma"] = 0.99
  config["lr"] = .0001
  config['num_gpus']=1

  apex_config = merge_dicts(
    config,  # see also the options in dqn.py, which are also supported
    {
        'rollout_fragment_length':16,
        "framework": "torch",
          "ignore_worker_failures": True,
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

      },

)
  

  apex_trainer = ApexTrainer(
      env="gfootball",
      config=apex_config)


for i in range(100000):

    # results = []
    episode_data = []
    print("+++++++++++++++Training iteration {!s}+++++++++++++++++++".format(i+1))
    result = apex_trainer.train()
    print(pretty_print(result))

    if ((i > 0) and (i % 10 == 0)):
        print("===============Swapping weights===================")
        #Store weights - Weights are stored as ordered dicts
        P4key_P3val = {} # temp storage with "policy_4" keys & "policy_3" values
        for (k4,v4), (k3,v3) in zip(apex_trainer.get_policy("policy_04").get_weights().items(),
                                    apex_trainer.get_policy("policy_03").get_weights().items()):
            P4key_P3val[k4] = v3
            #print("v3",v3)
            #print("P4key_P3val",P4key_P3val)

        P3key_P2val = {} # temp storage with "policy_3" keys & "policy_2" values
        for (k3,v3), (k2,v2) in zip(apex_trainer.get_policy("policy_03").get_weights().items(),
                                    apex_trainer.get_policy("policy_02").get_weights().items()):

            P3key_P2val[k3] = v2
            #print("v2",v2)
            #print("P3key_P2val",P3key_P2val)

        P2key_P1val = {} # temp storage with "policy_2" keys & "policy_1" values
        for (k2,v2), (k1,v1) in zip(apex_trainer.get_policy("policy_02").get_weights().items(),
                                    apex_trainer.get_policy("policy_01").get_weights().items()):

            P2key_P1val[k2] = v1
            #print("v1",v1)
            #print("P2key_P1val",P2key_P1val)

        #Set weights
        apex_trainer.set_weights({"policy_04":P4key_P3val, # weights or values from "policy_03" with "policy_04" keys
                                    "policy_03":P3key_P2val, # weights or values from "policy_02" with "policy_03" keys
                                    "policy_02":P2key_P1val, # weights or values from "policy_01" with "policy_02" keys
                                    "policy_01":apex_trainer.get_policy("policy_01").get_weights() # no change
                                })
        # To check
        for (k,v), (k2,v2) in zip(apex_trainer.get_policy("policy_01").get_weights().items(),
                                    apex_trainer.get_policy("policy_02").get_weights().items()):
        
            print("Check weights have been swapped")
            #print(v)
            #print(v2)
            assert (v == v2).all()
        
    if i % 200 == 0:
        checkpoint = apex_trainer.save()
        print("checkpoint saved at", checkpoint)
