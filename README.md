# GRF-Self-Play

This repository contains the implementation for competitive self-play with Google Research Football. Proximal Policy Optimisation is used to train the agent in a 11vs11 game using competitive self-play. 



<p align="center">
<img src="https://github.com/shehrum/GRF-Self-Play/blob/master/images/game.png" alt="game" width="600" height="400">
</p>
### Getting Started

Follow the installation instructions for [Google Research Football environment](https://github.com/google-research/football). After the environment is installed, clone this repository inside the the football environment folder.  


### Dependencies
Use the `requirements.txt` to install the required dependencies. 
```
pip install -r requirements.txt
```

### Instructions
1. You can run the training code for PPO using the following command in your terminal:

```
python train_ppo.py

```
This will start your model training, and save the results and the trained model in /ray_results.
Modify the file for parameter tuning.

2. For evaluation, run the following command after adding the checkpoint path in the code file.

```
python eval.py

```


### Credits
I would like to thank [Joshua Johnson](https://github.com/josjo80) for help with the self-play framework code, and [Anton Raichuk](https://research.google/people/AntonRaichuk/) for support in understanding the Google Research Football environment.
