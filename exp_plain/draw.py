import torch
import numpy as np
import random
import os
import yaml
import argparse
import sys
import gc
import time 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.LSH_proj_extra import SuperBitLSH
from _utils_.poison_loader import PoisonLoader
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import *

plot_single_curve_from_file("./results/poison_with_detection_lenet5_mnist_layers_proj_detect_label_flip_p0.20_IID.npz", None, "./results/poison_with_detection_lenet5_mnist_layers_proj_detect_label_flip_p0.20_IID.png")