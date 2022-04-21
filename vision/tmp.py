## This is a test file to train to track an object
## Author : Avadesh Meduri
## Date : 8/04/2022

from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision.io import read_image

import pickle

import cv2

fname = "data1"
img_dir = "./image_data/" + fname
data = np.load("position_data/" + fname + ".npz")
y_train = data["position"]
ct = 0
image = cv2.imread(img_dir + "/0.png", cv2.IMREAD_UNCHANGED )
image = read_image(img_dir + "/0.png", torchvision.io.ImageReadMode.UNCHANGED )