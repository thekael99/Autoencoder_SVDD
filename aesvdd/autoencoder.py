from tensorflow import keras


__all__ = ['mnist_lenet', 'cifar_lenet']


def mnist_lenet(H=32):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=(28, 28, 1)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False, name='code_layer'))

    return model


def cifar_lenet(H=128):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same', use_bias=False, input_shape=(32, 32, 3)))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(64, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    model.add(keras.layers.LeakyReLU(1e-2))
    model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(H, use_bias=False, name='code_layer'))

    return model


def Vanilla_AE(H=5):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(9,)))
    model.add(keras.layers.Dense(6, activation='relu', use_bias=False))
    model.add(keras.layers.Dense(H, activation='relu', use_bias=False, name='code_layer'))
    model.add(keras.layers.Dense(6, activation='relu', use_bias=False))
    model.add(keras.layers.Dense(9, activation='tanh', use_bias=False))
    return model


def VAE(H=5):
    pass