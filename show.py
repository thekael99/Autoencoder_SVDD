


import os
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

from dsvdd import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from sklearn.metrics import roc_auc_score, accuracy_score


def metrics_v1(y_test, y_predict):
    result = dict()
    result['acc'] = accuracy_score(y_test, y_predict)
    result['precision'] = precision_score(y_test, y_predict)
    result['Recall'] = recall_score(y_test, y_predict)
    result['F1'] = f1_score(y_test, y_predict)
    result['AUROC'] = roc_auc_score(y_test, y_predict)

    return result


def print_metric(y_test, y_predict):
    result = metrics_v1(y_test, y_predict)
    print('Accuracy         = %.4f ' % (result['acc']))
    print('Precision_score  = %.4f ' % (result['precision']))
    print('Recall_score     = %.4f ' % (result['Recall']))
    print('F1_score         = %.4f ' % (result['F1']))
    print('AUC              = %.4f ' % (result['AUROC']))


def Vanilla_AE(H=5):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(9,)))
    model.add(keras.layers.Dense(6, activation='relu', use_bias=False))
    model.add(keras.layers.Dense(H, activation='relu', use_bias=False, name='code_layer'))
    model.add(keras.layers.Dense(6, activation='relu', use_bias=False))
    model.add(keras.layers.Dense(9, activation='tanh', use_bias=False))
    return model


def get_paysim():
    df = pd.read_csv('D:\SourceTree\THESIS_2020-2021_CODE\AESVDD\paysim_dataset\paysim.csv')

    df['hour'] = df['step'] % 24
    df['day'] = (df['step'] / 24 ).apply(np.ceil)
    df['sin_hour'] = np.sin(2*np.pi*df.hour/24)
    df['type'] = LabelEncoder().fit_transform(df['type'])
    df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])
    
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest']
    df[features] = StandardScaler().fit_transform(df.loc[:, features].values)
    
    features_train = ['nameDest', 'oldbalanceOrg', 'newbalanceOrig', 'type', 'amount', 'sin_hour', 'day', 'step', 'hour']
    targets = ['isFraud']
    X = df.loc[:, features_train].values
    y = df.loc[:, targets].values
    X = X - (X.max(0)+X.min(0))/2
    X = X/X.max(axis=0) 
    y = y.reshape(-1,)
    y = np.where(y == 1, -1, 1)
    return X, y

    # df['hour'] = df['step'] % 24
    # df['day'] = (df['step'] / 24 ).apply(np.ceil)
    # df['weekday'] = df['day'] % 7
    # df['weekday'] = df['weekday'].replace({0.0 : 7.0})
    # df['sin_hour'] = np.sin(2*np.pi*df.hour/24)
    # df['cos_hour'] = np.cos(2*np.pi*df.hour/24)
    # df['sin_weekday'] = np.sin(2*np.pi*df.weekday/7)
    # df['cos_weekday'] = np.cos(2*np.pi*df.weekday/7)
    # df['type'] = LabelEncoder().fit_transform(df['type'])
    # df['nameOrig'] = LabelEncoder().fit_transform(df['nameOrig'])
    # df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])
    
    # features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig', 'nameDest']
    # features_values = df.loc[:, features].values
    # scaler_values = StandardScaler().fit_transform(features_values)
    
    # features_train = ['nameDest', 'oldbalanceOrg', 'newbalanceOrig', 'type', 'amount', 'sin_hour', 'day', 'step', 'hour']

    # targets = ['isFraud']
    # X = df.loc[:, features_train].values
    
    # X = X - (X.max(0)+X.min(0))/2
    # X = X/X.max(axis=0) 
    
    # y = df.loc[:, targets].values
    # y = y.reshape(-1,)
    # y = np.where(y == 1, -1, 1)
    # return X, y

# In[4]:

print('-----------   start get dataset     -----------')
X, y = get_paysim()
print('-----------   get dataset  Sucessful -----------')


# In[5]:


# #ratio train-test 0.7 0.3
# X_train, X_test , y_train, y_test = train_test_split(X, y, train_size=0.6)

#ratio train-val-test 0.6 0.2 0.2
# X_train, X_test , y_train, y_test = train_test_split(X, y, train_size=800000, test_size=200000)
# X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, train_size=0.75)

X_train, X_test , y_train, y_test = train_test_split(X, y, train_size=0.8)
X_train, X_val , y_train, y_val = train_test_split(X_train, y_train, train_size=0.75)


tf.reset_default_graph()
AE_model = Vanilla_AE(H=5)


#pre_train_AE (not required)
AE_model.compile(optimizer='adam', loss='mse')
print("pre-train AE")
AE_model.fit(X_train, X_train, epochs=3, batch_size=128, validation_data=(X_test, X_test), verbose=1)


inputs = AE_model.input
outputs = AE_model.get_layer(name="code_layer").output
encoder = Model(inputs = inputs, outputs = outputs)


# build model and DeepSVDD
svdd_soft = DeepSVDD(encoder, input_shape=(9,), representation_dim=5, objective='hard', nu=0.05, batch_size=128, lr=1e-4)

t0 = time.time()
hisory_soft = svdd_soft.fit(X_train, X_val, y_val, epochs=20, verbose=True)
print('Train time:', time.time() - t0)


print('-------------------------------')
print('-------------------------------')
# test DeepSVDD
t0 = time.time()
score = svdd_soft.predict(X_test)
auc = roc_auc_score(y_test, -score)
print('Test time:', time.time() - t0)
print("---------------- Dataset---------------")
print("Train_set size:   ", len(X_train))
print("Test_set size:    ", len(X_test))
print("nu:    ", svdd_soft.nu)
print('AUROC: %.4f' % auc)


label = svdd_soft.predict_label(X_test)
auc = roc_auc_score(y_test, label)
acc = accuracy_score(y_test, label)
print('AUROC via Label: %.4f' % auc)
print('ACC via Label: %.4f' % acc)



# In[ ]:

print('-------------------------------')
print('-------------------------------')


# from sklearn.svm import OneClassSVM
# from sklearn.metrics import accuracy_score, recall_score,precision_score, f1_score, roc_curve, auc
# import matplotlib.pyplot as plt

# #clf = OneClassSVM()
# start = time.time()
# model = OneClassSVM(nu=0.05, gamma='auto', kernel='rbf')
# # scoring = clf.fit(trainData_np)
# print("---------------- Dataset---------------")
# print("Train_set size:   ", len(X_train))
# print("Test_set size:    ", len(X_test))
# print("nu:    ", model.nu)
# print("---------------- Start Training---------------")
# model.fit(X_train)
# #nu=0.261, gamma=0.05
# stop = time.time()
# print("---------------- Finish Training---------------")
# print(f"Training time: {stop - start}s")

# print('-----------------------------------------------')
# print('---------------- Test Result -----------------')
# y_predict_origin_data = model.predict(X_test)
# print_metric(y_test, y_predict_origin_data)