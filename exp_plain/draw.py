from model.Lenet5 import LeNet5
from model.Cifar10Net import CIFAR10Net
from _utils_.LSH_proj_extra import SuperBitLSH
from _utils_.poison_loader import PoisonLoader
from defence.score import ScoreCalculator
from defence.kickout import KickoutManager
from _utils_.dataloader import load_and_split_dataset
from _utils_.save_config import *

plot_single_curve_from_file("./results/poison_no_detection_cifar10_cifar10_none_['model_compress']_0.1_False.npz", None, "./results/poison_no_detection_cifar10_cifar10_none_['model_compress']_0.1_False.png")