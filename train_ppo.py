
"""Script for training agent in Google Research Football with competitive self-play. 
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
from ray.rllib.agents.ppo.ppo import PPOTrainer

import ray.rllib.agents.ppo as ppo
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
  ray.init(num_gpus=1)

  # Register model
  ModelCatalog.register_custom_model("my_model", CustomTorchModel)
  # Register environment
  register_env('gfootball', lambda _: RllibGFootball({"mode":"train"}))
  # Get single environment instance
  single_env = RllibGFootball({"mode":"train"})
  obs_space = single_env.observation_space
  act_space = single_env.action_space


# Defining configuration of PPO trainer
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
    'observation_filter': 'NoFilter',
          # "_fake_gpus": True,
          "num_workers": 8,
          "num_envs_per_worker": 4,
          "num_gpus": 1, #Driver GPU
          # "num_gpus_per_worker": 0.05, #GPU/worker 
          "ignore_worker_failures": True,
          "train_batch_size": 6400,
          # 'sample_batch_size': 100,
          'rollout_fragment_length': 200, #350, #512, #200, 300
          "sgd_minibatch_size": 500,
          # 'num_sgd_iter': 10,
          "lr": 2.5e-4, #2.5e-4, #2.5e-3, #2.5e-4, # 3e-4,
          "lambda": .95,
          "gamma": .998,
          "entropy_coeff": 1e-4, # try 0.01
          "kl_coeff": 1.0,
          "clip_param": 0.2,
          "num_sgd_iter": 10,
          "vf_share_layers": True, # share layers
          "vf_clip_param": 10.0, 
          "vf_loss_coeff": 10, #1.0,0.5 #Tune this to scale the loss
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
# Uncomment to turn on evaluation intervals while training
        # "evaluation_interval": 50,
        # "evaluation_config": {
        #   "env_config": {
        #     # Use test set to evaluate
        #     'mode': 'test'
        #   },
        #     "explore": False,
        #     "multiagent": {
        #         "policies": {
        #             # the first tuple value is None -> uses default policy class
        #             "policy_01": (None, obs_space, act_space, {}),
        #             "policy_02": (None, obs_space, act_space, {}),
        #             "policy_03": (None, obs_space, act_space, {}),
        #             "policy_04": (None, obs_space, act_space, {}),
        #         },
        #         "policy_mapping_fn": policy_mapping_fn_eval
        #     },
        # },

      })

# Uncomment to restore from check-point
# ppo_trainer.restore('/home/ubuntu/ray_results/PPO_gfootball_2020-08-05_20-46-06e4wbrrqg/checkpoint_301/checkpoint-301')
for i in range(100000):

    episode_data = []
    print("+++++++++++++++Training iteration {!s}+++++++++++++++++++".format(i+1))
    result = ppo_trainer.train()
    print(pretty_print(result))

    if ((i > 0) and (i % 10 == 0)):
        print("===============Swapping weights===================")

        P4key_P3val = {} # temp storage with "policy_4" keys & "policy_3" values
        for (k4,v4), (k3,v3) in zip(ppo_trainer.get_policy("policy_04").get_weights().items(),
                                    ppo_trainer.get_policy("policy_03").get_weights().items()):
            P4key_P3val[k4] = v3


        P3key_P2val = {} # temp storage with "policy_3" keys & "policy_2" values
        for (k3,v3), (k2,v2) in zip(ppo_trainer.get_policy("policy_03").get_weights().items(),
                                    ppo_trainer.get_policy("policy_02").get_weights().items()):

            P3key_P2val[k3] = v2


        P2key_P1val = {} # temp storage with "policy_2" keys & "policy_1" values
        for (k2,v2), (k1,v1) in zip(ppo_trainer.get_policy("policy_02").get_weights().items(),
                                    ppo_trainer.get_policy("policy_01").get_weights().items()):

            P2key_P1val[k2] = v1

        #Set weights of policies
        ppo_trainer.set_weights(
                                    "policy_04":P4key_P3val, # weights or values from "policy_03" with "policy_04" keys
                                    "policy_03":P3key_P2val, # weights or values from "policy_02" with "policy_03" keys
                                    "policy_02":P2key_P1val, # weights or values from "policy_01" with "policy_02" keys
                                    "policy_01":ppo_trainer.get_policy("policy_01").get_weights() # no change
                                })
        # To check
        for (k,v), (k2,v2) in zip(ppo_trainer.get_policy("policy_01").get_weights().items(),
                                    ppo_trainer.get_policy("policy_02").get_weights().items()):
        
            print("Check weights have been swapped")
            assert (v == v2).all()
        
    if i % 200 == 0:
      # Save checkpoint
        checkpoint = ppo_trainer.save()
        print("checkpoint saved at", checkpoint)
