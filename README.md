# GymValueIterationNetwork
# Value Iteration Networks - GYM Environment #

This project is extension of the work of Aviv T. et. al. (https://arxiv.org/abs/1602.02867), by applying VIN model to Open AI gym environment - LunarLander-v2 and BipedalWalker-v2.

## Dependencies ##
Python 3.6  
Numpy  
Matplotlib  
Opencv  
Pytorch  
TensorFlow  
Openai GYM  
Openai Baseline  
Visdom  

clone all files with:  
```
$ git clone https://github.com/LiorAl/GymValueIterationNetworks
```
## Train ##
In order to train LunarLander use:
```
$ python main.py --env-name "LunarLander-v2" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 6 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01
```

In order to train BipedalWalker use:
```
$ python main.py --env-name "BipedalWalker-v2" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 6 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1 --use-linear-lr-decay --use-linear-clip-decay --entropy-coef 0.01
```

## Test ##
In order to load pretrain model and run LunarLander use:
```
$ python test_agent.py --load-dir trained_models/ppo --env-name "LunarLander-v2"
```
BipedalWalker:
```
python test_agent.py --load-dir trained_models/ppo --env-name "BipedalWalker-v2"
```
## Files Description ##
**ppo** -> files for PPO algorithm.  
**trained_models** -> Pretrained models.  
**BipedalModel.py** -> VIN model for BpidealWalker env.  
**LunarLanderModel.py** -> VIN model for LunarLander env.  
**main.py** -> main script for training VIN architecture.  
