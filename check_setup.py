complete = True

try:
    import os
    import numpy
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import ray
    from datetime import datetime
    import pytz
    import pandas
    import gym
    import gtimer
    import pygame
    print("Success: All Python packages are available!")
except ImportError as e:
    complete = False
    print("Error: Could not import all packages!")
    print(e)

try:
    import smrl
    import mrl_analysis
    import rlkit
    import symmetrizer
    import meta_envs
    print("Success: All local packages are available!")
except ImportError as e:
    complete = False
    print("Error: Could not import all local packages!")
    print(e)

try:
    import mujoco_py
    print("Success: All optional packages are available!")
except ImportError as e:
    print("Optional packages might be missing (that's fine...)")


if complete:
    print("Setup is complete.")