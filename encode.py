from __future__ import print_function
import os
import pickle

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE, order_sim
from collections import OrderedDict

from vocab import Vocabulary
import evaluation

modelpath = "runs/coco_vse++_resnet_restval_finetune/model_best.pth.tar"
datapath = "data/"
evaluation.evalrank(modelpath, data_path=datapath, split="val")