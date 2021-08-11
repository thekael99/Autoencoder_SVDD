import sys
import os
import time
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from aesvdd import autoencoder
from aesvdd import model
from aesvdd import datasets

import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(os.getcwd())
warnings.filterwarnings("ignore")
sys.path.append(".")



import pyarrow.feather as feather
import pickle
import tensorflow.compat.v1 as tf
from tensorflow.keras import Model
from tensorflow import keras
import time
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from MinMaxNormalize import MinMax
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from flask_restful import Api, Resource, reqparse
from flask import Flask, request


CONST1 = np.array([0.5,   0.5,   0.5,   2.,   0.5,   0.,  16., 372.,  11.5])
CONST2 = np.array([1.,   1.,   1.,   4.,   1.,   1.,  31., 743.,  23.])

class MODEL(Resource):

    def handle_request_body(args):
        data = []
        data.append(MinMax().nameDest(args['nameDest_code']))
        data.append(MinMax().oldbalanceOrg(args['oldBalanceOrig']))
        data.append(MinMax().newbalanceOrig(args['newBalanceOrig']))
        data.append(args['type'])
        data.append(MinMax().amount(args['amount']))
        data.append(0)
        data.append(0)
        data.append(args['step'])
        data.append(0)

        df = pd.DataFrame([data], columns=[
            'nameDest_code', 'oldbalanceOrg', 'newbalanceOrig', 'type', 'amount', 'sin_hour', 'day', 'step', 'hour'])
        df['hour'] = df['step'] % 24
        df['day'] = (df['step'] / 24).apply(np.ceil)
        df['sin_hour'] = np.sin(2*np.pi*df.hour/24)
        return (df.head(1).to_numpy() - CONST1)/CONST2

    def post(self, action="predict"):

        encoder = keras.models.load_model("./checkpoints/0_encoder.h5")
        c = np.fromfile("./checkpoints/0_c")
        hard_object = model.HARD(encoder=encoder, c=c)

        parser = reqparse.RequestParser()
        parser.add_argument('step', type=int)
        parser.add_argument('type', type=int)
        parser.add_argument('nameDest_code', type=int)
        parser.add_argument('amount', type=float)
        parser.add_argument('oldBalanceOrig', type=float)
        parser.add_argument('newBalanceOrig', type=float)
        args = parser.parse_args()
        data = MODEL.handle_request_body(args)

        out = {'result': str(hard_object.predict_label(data)[0])}
        return out, 200




APP = Flask(__name__)
API = Api(APP)
API.add_resource(MODEL, '/api/<action>')



if __name__ == '__main__':

    APP.run(debug=True, port='5000')
