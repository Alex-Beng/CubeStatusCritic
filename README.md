# Cube Status Critic

English | [中文](./README_CN.md)


Using IRL(Inverse Reinforcement Learning) to train a reward function for a Rubik's cube/other cube's status.

# Usage

## Collect human prefer data throw custom [cstimer](https://alex-beng.github.io/cstimer/)

![set prefer](./pics/1.png)

When solved a scramble, click `prefer` to switch the preference against last scramble, or use shortcut `Ctrl + J` or `Ctrl + K` to set prefer or not.


There while be an up arrow besied the result to showing the preference to this scramble against the last one.


All the timing ways should be supported, if you got any problems in using custom cstimer, please issue [here](https://github.com/Alex-Beng/CubeStatusCritic/issues).

## Export data

![export data](./pics/2.png)

Use cstimer's `Export to file` built-in function to export the data file, which will be used in the traing part.


## Write config in `config.py` and train

Make a `config.py` firstly like:
```bash
cp config.py.template config.py
```

Custom your config in the `config.py`


## Train, export and deployment ONNX model

TODO