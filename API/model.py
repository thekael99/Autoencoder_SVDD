import sys
import os
import joblib
import scipy.io as sio
import numpy as np
from dataset.paysim_dataset import load_paysim_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from dsvdd import *
from sklearn.metrics import roc_auc_score


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


def train():
    # load Paysim dataset
    print("---------------- Start get Dataset ---------------")
    X, y = get_paysim()
    print("---------------- Get Dataset: Sucessful ---------------")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=80000, test_size=20000)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25)

    model = OneClassSVM(nu=0.261, gamma='auto', kernel='rbf')
    for i in range(0, 5):
        print('Epochs:  ', i)
        print("---------------- Start Training---------------")
        model.fit(X_train)
        print("---------------- Training  Sucessful ---------------")
        print('------------Validation Result-------------')
        y_predict_origin_data = model.predict(X_val)
        auc = roc_auc_score(y_val, y_predict_origin_data)
        print('AUC:  ', auc)
    return model


if __name__ == '__main__':

    mdl = train()
    joblib.dump(mdl, 'aesvdd.mdl')
