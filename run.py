import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(os.getcwd())

import time
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings("ignore")

from aesvdd import networks
from aesvdd import AESVDD
from aesvdd import datasets

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.metrics import roc_auc_score

"""
Get option:
    --dataset:
    --model autoencoder:
    --objective: hard or soft
    --epoch pretrain:
    --epoch train:
    --nu: 0.05
    --lr: 5e-4
    --verbose: True
    --another...
"""
dataset = sys.argv[1]
aemodel = sys.argv[2]
objective = sys.argv[3]
epoch_pretrain = int(sys.argv[4])
epoch_train = int(sys.argv[5])
nu = float(sys.argv[6])
lr = float(sys.argv[7])
verbose = bool(sys.argv[7])

if dataset
X_train, X_test, y_train, y_test = datasets.get_paysim(train_ratio=0.7)

tf.reset_default_graph()
AE_model = networks.Vanilla_AE(H=5)

#pre_train_AE (not required)
if epoch_pretrain > 0:
    AE_model.compile(optimizer='adam', loss='mse')
    AE_model.fit(X_train, X_train, epochs=epoch_pretrain, batch_size=128, validation_data=(X_test, X_test), verbose=verbose)


inputs = AE_model.input
outputs = AE_model.get_layer(name="code_layer").output
encoder = Model(inputs = inputs, outputs = outputs)


# build model and DeepSVDD
ae_svdd = AESVDD.AESVDD(encoder, input_shape=(9,), representation_dim=5, objective=objective, nu=nu)

t0 = time.time()
hisory = svdd_soft.fit(X_train, X_train, y_train, epochs=epoch_train, verbose=verbose)
print('Train time:', time.time() - t0)

# test DeepSVDD
t0 = time.time()
score = svdd_soft.predict(X_test)
auc = roc_auc_score(y_test, -score)
print('Test time:', time.time() - t0)
print('------------- AUROC ----------- : %.4f' % auc)
