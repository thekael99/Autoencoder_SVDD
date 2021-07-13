from aesvdd import datasets
from aesvdd import model
from aesvdd import autoencoder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import os

from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


def main(cls=1):
    tf.reset_default_graph()
    # from aesvdd.utils import plot_most_normal_and_abnormal_images
    # build model and DeepSVDD
    keras_model = autoencoder.mnist_lenet(32)
    svdd = model.AESVDD(keras_model, input_shape=(28, 28, 1), representation_dim=32, objective='hard')

    # get dataset
    X_train, X_test, y_test = datasets.get_mnist(cls)
    # X_train, X_test, y_test = get_cifar10(cls)
    print(X_train.shape)

    # train DeepSVDD
    svdd.fit(X_train, X_test, y_test, epochs=5, verbose=True)

    # test DeepSVDD
    score = svdd.predict(X_test)
    auc = roc_auc_score(y_test, -score)
    print('AUROC: %.3f' % auc)

if __name__ == '__main__':
    main(1)
