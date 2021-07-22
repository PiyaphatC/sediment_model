import sys
sys.path.append('../')
from hydroDL import master, utils
from hydroDL.master import default
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat


import numpy as np
import os
import torch
import pandas as pd
import math
import random

t = np.arange(60).reshape(3,4,5)
f = np.load('/data/pmc5570/Sediment_model/Sed/FirstRun/epochs500_batch60_rho365_hiddensize200_Tstart19800101_Tend20101231/All-1980-2010/x.npy')
S = (f[:, :, 0])

l = 'park'


predLst = np.load('/data/pmc5570/Sediment_model/Sed/FirstRun/epochs500_batch60_rho365_hiddensize200_Tstart19800101_Tend20101231/All-1980-2010/pred.npy')
obsLst = np.load('/data/pmc5570/Sediment_model/Sed/FirstRun/epochs500_batch60_rho365_hiddensize200_Tstart19800101_Tend20101231/All-1980-2010/obs.npy')

statDictLst = [stat.statError(x.squeeze(), y.squeeze()) for (x, y) in zip(predLst, obsLst)]

l = 'park'