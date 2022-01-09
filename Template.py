from Networks import Networks
from Training import Training

import torch.nn as nn
import numpy as np
import torch
import os


# =============================================================================
# Prerequisites
# =============================================================================
#Paths to the data and save locations
trainImage = os.path.abspath(os.curdir + "/data/monet_jpg/")
trainPaint = os.path.abspath(os.curdir + "/data/photo_jpg/")

#Final Destination has to be changed and Filename
modelpath = '/home/Kaggle_Challenges/ImSomethingOfAPainterMyself/'  ###
filename = 'Adriana_FirstTry' ###

trainPaths = {'Imagepath': trainImage, 'Paintingpath': trainPaint, 'Modelpath': modelpath, 'Filename': filename}

#Training and Network
Epoch_BatchSize = np.array([1, 1, 2]) ###

Generator = Networks.Adriana(3, 3)
Discriminator = Networks.Candice(3, 1) ###
loss_fn = nn.BCEWithLogitsLoss() ###     #If Not CEL => Softmax = True
optimizerG = torch.optim.Adam(Generator.parameters(), lr=2e-4, weight_decay=3e-05, betas=(0.5,0.999)) ###
optimizerD = torch.optim.Adam(Discriminator.parameters(), lr=2e-4, weight_decay=3e-05, betas=(0.5, 0.999)) ###

trainClass = Training.TrainGAN(trainPaths, Generator, Discriminator, loss_fn, Epoch_BatchSize)       
trainClass.trainModel(optimizerD, optimizerG)                          

