from .utils import task
from tqdm import tqdm
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from math import ceil
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.compat.v1.keras.backend as K



class AESVDD:
    def __init__(self, keras_model, input_shape=(28, 28, 1), objective='hard',
                 nu=0.1, representation_dim=32, batch_size=128, lr=1e-3):
        self.represetation_dim = representation_dim
        self.objective = objective
        self.keras_model = keras_model
        self.nu = nu
        self.R = tf.get_variable('R', [], dtype=tf.float32, trainable=False)
        self.c = tf.get_variable('c', [self.represetation_dim], dtype=tf.float32, trainable=False)
        self.warm_up_n_epochs = 10
        self.batch_size = batch_size

        with task('Build graph'):
            self.x = tf.placeholder(tf.float32, [None] + list(input_shape))
#             print(self.x.shape)
            self.latent_op = self.keras_model(self.x)
#             print(self.latent_op.shape)
            self.dist_op = tf.reduce_sum(tf.square(self.latent_op - self.c), axis=-1)

            if self.objective == 'soft':
                self.score_op = self.dist_op - self.R ** 2
                penalty = tf.maximum(self.score_op, tf.zeros_like(self.score_op))
                self.loss_op = self.R ** 2 + (1 / self.nu) * penalty

            else:  # one-class
                self.score_op = self.dist_op
                self.loss_op = self.score_op

            opt = tf.train.AdamOptimizer(lr)
            self.train_op = opt.minimize(self.loss_op)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        self.sess.close()

    def fit(self, X, X_test, y_test, epochs=10, verbose=True):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))

        self.sess.run(tf.global_variables_initializer())
        self._init_c(X)

        ops = {
            'train': self.train_op,
            'loss': tf.reduce_mean(self.loss_op),
            'dist': self.dist_op
        }
        keras.backend.set_learning_phase(True)

        for i_epoch in range(epochs):
            ind = np.random.permutation(N)
            x_train = X[ind]
            g_batch = tqdm(range(BN)) if verbose else range(BN)
            for i_batch in g_batch:
                x_batch = x_train[i_batch * BS: (i_batch + 1) * BS]
                results = self.sess.run(ops, feed_dict={self.x: x_batch})

                if self.objective == 'soft' and i_epoch >= self.warm_up_n_epochs:
                    self.sess.run(tf.assign(self.R, self._get_R(results['dist'], self.nu)))

            else:
                if verbose:
                    pred = self.predict(X_test)  # pred: large->fail small->pass
                    auc = roc_auc_score(y_test, -pred)  # y_test: 1->pass 0->fail
                    print('\rEpoch: %3d AUROC: %.3f' % (i_epoch, auc))

    def predict(self, X):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        scores = list()
        keras.backend.set_learning_phase(False)

        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            s_batch = self.sess.run(self.score_op, feed_dict={self.x: x_batch})
            scores.append(s_batch)

        return np.concatenate(scores)

    def predict_label(self, X):
        score = self.predict(X)
        if self.objective == "hard":
            score_temp = np.msort(score)
            threshold = score_temp[score.size - (int)(score.size * self.nu)]
            return np.where(score > threshold, -1, 1)
        else:
            return np.where(score > 0, -1, 1)

    def _init_c(self, X, eps=1e-1):
        N = X.shape[0]
        BS = self.batch_size
        BN = int(ceil(N / BS))
        keras.backend.set_learning_phase(False)

        with task('1. Get output'):
            latent_sum = np.zeros(self.latent_op.shape[-1])
            for i_batch in range(BN):
                x_batch = X[i_batch * BS: (i_batch + 1) * BS]
                latent_v = self.sess.run(self.latent_op, feed_dict={self.x: x_batch})
                latent_sum += latent_v.sum(axis=0)

            c = latent_sum / N

        with task('2. Modify eps'):
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

        self.sess.run(tf.assign(self.c, c))
        self.c_np = c 

    def _get_R(self, dist, nu):
        return np.quantile(np.sqrt(dist), 1 - nu)

    def save_model(self, model_name):
        path = "./checkpoints/" + model_name 
        self.keras_model.save(path + "_encoder.h5")
        self.c_np.tofile(path + "_c")

    def _load_c(self, c):
        self.sess.run(tf.assign(self.c, c))
    
    def _load_encoder(self, encoder):
        self.keras_model = encoder
        self.latent_op = self.keras_model(self.x)


class HARD:
    def __init__(self, encoder, c, nu = 0.05):
        self.encoder = encoder
        self.c = c
        self.nu = nu
    
    def predict(self, X, batch_size = 128):
        N = X.shape[0]
        BS = batch_size
        BN = int(ceil(N / BS))
        scores = list()

        for i_batch in range(BN):
            x_batch = X[i_batch * BS: (i_batch + 1) * BS]
            s_batch = np.sum(np.square(self.encoder.predict(x_batch)-self.c), axis=1)
            scores.append(s_batch)

        return np.concatenate(scores) 

    def predict_label(self, X):
        score = self.predict(X)
        score_temp = np.msort(score)
        threshold = score_temp[score.size - (int)(score.size * self.nu)]
        return np.where(score > threshold, -1, 1)
    
    def evalute(self, X_test, y_test):
        pred = self.predict(X_test)
        auc = roc_auc_score(y_test, -pred)
        return auc