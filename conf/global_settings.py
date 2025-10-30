""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

LEFTEYE_TRAIN_PATH = "/home/zhanghangning/pytorch/left-eye/train"
LEFTEYE_TEST_PATH = "/home/zhanghangning/pytorch/left-eye/test"

LEFTEYE_TEST_MEAN = (0.5813, 0.4787, 0.4483)
LEFTEYE_TEST_STD = (0.1629, 0.1960, 0.2080)

LEFTEYE_TRAIN_MEAN = (0.5818, 0.4788, 0.4483)
LEFTEYE_TRAIN_STD = (0.1630, 0.1961, 0.2080)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 100
MILESTONES = [30, 60, 80]  # 训练 100 轮，总体衰减三次

#initial learning rate
INIT_LR = 0.003

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








