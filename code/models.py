"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, UpSampling2D, BatchNormalization, ReLU

import hyperparameters as hp


class CNNModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(CNNModel, self).__init__()

        self.optimizer = tf.keras.optimizers.SGD()

        self.architecture = [
            Conv2D(3, 3, 1, activation="relu", padding="same"),
            MaxPool2D(2, padding="same"),
            UpSampling2D(size=(2, 2))
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    # @staticmethod
    # def loss_fn(labels, predictions):
    #     """ Loss function for the model. """
    #     #TODO: find new loss function
    #     return tf.keras.losses.MeanSquaredError(labels, predictions)


class RESCNNModel(tf.keras.Model):
    """ CNN that uses ResNet """

    def __init__(self):
        super(RESCNNModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.RES = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_shape=(hp.img_size, hp.img_size, 3))

        for layer in self.RES.layers:
            layer.trainable = False

        self.head = [
            # input size: (4, 4, 2048)
            # Dropout(0.4),
            Conv2D(256, 3, padding="same"),
            BatchNormalization(),
            ReLU(),
            UpSampling2D(size=(7, 7)),
            # (28, 28, 256)
            Conv2D(128, 3, padding="same"),
            BatchNormalization(),
            ReLU(),
            UpSampling2D(size=(2, 2)),
            # (56, 56, 128)
            Conv2D(64, 3, padding="same"),
            BatchNormalization(),
            ReLU(),
            UpSampling2D(size=(2, 2)),
            # (112, 112, 64)
            Conv2D(32, 3, padding="same"),
            BatchNormalization(),
            ReLU(),
            # (112, 112, 32)
            Conv2D(2, 3, activation="tanh", padding="same"),
            # (112, 112, 2)
        ]

        self.model = tf.keras.Sequential(name="RES")
        self.model.add(self.RES)
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes input image through the network. """
        x = tf.concat((x,)*3, axis=-1)
        x = self.model(x)
        x = self.head(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        # TODO: find new loss function
        return tf.keras.losses.MeanSquaredError(labels, predictions)


class GANModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(GANModel, self).__init__()

        self.optimizer = tf.keras.optimizers.SGD()

        self.architecture = [
            Conv2D(32, 3, 1, activation="relu",
                   padding="same"), MaxPool2D(2, padding="same")
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        # TODO: find new loss function
        return tf.keras.losses.MeanSquaredError(labels, predictions)
