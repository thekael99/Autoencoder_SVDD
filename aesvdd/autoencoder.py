from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Input, Dense, Lambda, Conv2D, LeakyReLU, BatchNormalization, MaxPool2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy

__all__ = ['mnist_lenet', 'cifar_lenet', 'Vanilla_AE', 'VAE']


def mnist_lenet(H=32):
    model = Sequential()

    model.add(Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(1e-2))
    model.add(BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(MaxPool2D())

    model.add(Conv2D(4, (5, 5), padding='same', use_bias=False))
    model.add(LeakyReLU(1e-2))
    model.add(BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(H, use_bias=False, name='code_layer'))

    return model


def cifar_lenet(H=128):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), strides=(3, 3), padding='same', use_bias=False, input_shape=(32, 32, 3)))
    model.add(LeakyReLU(1e-2))
    model.add(BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(Conv2D(64, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(1e-2))
    model.add(BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(Conv2D(128, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(1e-2))
    model.add(BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(Flatten())
    model.add(Dense(H, use_bias=False, name='code_layer'))

    return model


def Vanilla_AE(H=5):
    model = Sequential()
    model.add(InputLayer(input_shape=(9,)))
    model.add(Dense(6, activation='relu', use_bias=False))
    model.add(Dense(H, activation='relu', use_bias=False, name='code_layer'))
    model.add(Dense(6, activation='relu', use_bias=False))
    model.add(Dense(9, activation='tanh', use_bias=False))
    return model


def VAE(z_dim=5):
    # encoder
    x = Input(shape=(9,))
    x_encoded1 = Dense(8, activation='relu', use_bias=False)(x)
    # x_encoded2 = Dense(6, activation='relu', use_bias=False)(x_encoded1)

    mu = Dense(z_dim, name='code_layer')(x_encoded1)
    log_var = Dense(z_dim)(x_encoded1)

    # sampling function
    batch_size = 100
    def sampling(args):
        mu, log_var = args
        eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
        return mu + K.exp(log_var) * eps


    z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

    # decoder
    z_decoder = Dense(8, activation='relu', use_bias=False)    
    y_decoder = Dense(9, activation='tanh', use_bias=False)

    z_decoded = z_decoder(z)
    y = y_decoder(z_decoded)

    #loss
    reconstruction_loss = BinaryCrossentropy()(x, y) * 9
    kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis=-1)
    vae_loss = reconstruction_loss + kl_loss
    
    #build
    model = Model(x, y)
    model.add_loss(vae_loss)
    model.compile(optimizer='rmsprop')

    return model
