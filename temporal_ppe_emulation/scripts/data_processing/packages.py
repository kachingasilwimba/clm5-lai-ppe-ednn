from mlguess.keras.models import EvidentialRegressorDNN
import keras
import numpy as np
from mlguess.keras.callbacks import ReportEpoch
from mlguess.keras.losses import evidential_cat_loss, evidential_reg_loss
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import xarray as xr
import glob
from os.path import join
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
# from dataloader import create_xy_data
#-----------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
#-----------------------------------
from SALib.analyze import fast as fast_analyze
from SALib.sample import fast_sampler
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.sample.sobol import sample as sobol_sample
import warnings
warnings.filterwarnings("ignore")