import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

__all__ = ['get_mnist', 'get_cifar10']


def get_mnist(cls=1):
    d_train, d_test = keras.datasets.mnist.load_data()
    x_train, y_train = d_train
    x_test, y_test = d_test

    mask = y_train == cls

    x_train = x_train[mask]
    x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
    x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)

    y_test = (y_test == cls).astype(np.float32)
    return x_train, x_test, y_test


def get_cifar10(cls=1):
    d_train, d_test = keras.datasets.cifar10.load_data()
    x_train, y_train = d_train
    x_test, y_test = d_test
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    mask = y_train == cls

    x_train = x_train[mask]
    x_train = (x_train / 255.).astype(np.float32)
    x_test = (x_test / 255.).astype(np.float32)

    y_test = (y_test == cls).astype(np.float32)
    return x_train, x_test, y_test


def get_paysim():
    df = pd.read_csv('./data/PS_20174392719_1491204439457_log.csv')
    df = pd.concat([df, pd.get_dummies(df['type'], prefix='type_')], axis=1)
    df['hour'] = df['step'] % 24
    df['day'] = (df['step'] / 24).apply(np.ceil)
    df['weekday'] = df['day'] % 7
    df['weekday'] = df['weekday'].replace({0.0: 7.0})
    df['sin_hour'] = np.sin(2*np.pi*df.hour/24)
    df['cos_hour'] = np.cos(2*np.pi*df.hour/24)
    df['sin_weekday'] = np.sin(2*np.pi*df.weekday/7)
    df['cos_weekday'] = np.cos(2*np.pi*df.weekday/7)
    df['type'] = LabelEncoder().fit_transform(df['type'])
    df['nameOrig'] = LabelEncoder().fit_transform(df['nameOrig'])
    df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])

    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig', 'nameDest']
    features_values = df.loc[:, features].values
    scaler_values = StandardScaler().fit_transform(features_values)
    df[features] = PCA(n_components=len(features)).fit_transform(scaler_values)

    features_train = ['nameDest', 'oldbalanceOrg', 'newbalanceOrig', 'type', 'amount', 'sin_hour', 'day', 'step', 'hour']

    targets = ['isFraud']
    X = df.loc[:, features_train].values

    X = X - (X.max(0)+X.min(0))/2
    X = X/X.max(axis=0)

    y = df.loc[:, targets].values
    y = y.reshape(-1,)
    y = np.where(y == 1, -1, 1)

    return X, y
