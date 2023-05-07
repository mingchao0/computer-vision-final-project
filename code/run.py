"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgba2rgb, rgb2gray, rgb2lab, lab2rgb
from skimage.io import imread
from tensorboard_utils import \
    ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import CNNModel, GANModel, RESCNNModel
from preprocess import Datasets
from skimage.transform import resize
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        default='cnn',
        choices=['cnn', 'cnn-pre', 'gan'],
        help='''Which model to use''')
    parser.add_argument(
        '--data',
        default='TODO',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    return parser.parse_args()


def thresholded_loss(y_true, y_pred):
    threshold = 10
    true_flatten = tf.reshape(y_true, (-1, 2))
    pred_flatten = tf.reshape(y_pred, (-1, 2))
    diff = pred_flatten - true_flatten
    dist = tf.math.sqrt(tf.math.add(tf.math.square(
        diff[:, 0]), tf.math.square(diff[:, 1])))
    successful_indices = tf.where(dist < threshold, 1, 0)
    y, indx, counts = tf.unique_with_counts(successful_indices)
    counter = 0
    index = 0
    for i in y:
        if i == 1:
            index = counter
        counter += 1
    num_success = counts[index]
    prop = num_success / len(true_flatten)
    return prop


def train(model, datasets, checkpoint_path, init_epoch):
    """ Training routine. """
    callback_list = [
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=logs_path,
        #     update_freq='batch',
        #     profile_batch=0),
        # ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.model, hp.max_num_weights)
    ]

    print("Fit model on training data")
    history = model.fit(
        datasets.train_L,
        datasets.train_ab,
        batch_size=64,
        epochs=hp.num_epochs,
        callbacks=callback_list,
        initial_epoch=init_epoch,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(datasets.val_L, datasets.val_ab),
    )

    print(history.history)


def test(model, datasets):
    """ Testing routine. """
    model.evaluate(
        x=datasets.test_L,
        y=datasets.test_ab,
        verbose=1,
    )
    #print(model, test_data)


def predict(model, datasets):
    l_test = datasets.test_L[:5]
    ab_test = datasets.test_ab[:5]

    ab_model = model.predict(l_test)

    for i in range(5):
        # Print just L
        L = l_test[i]
        grey_channel = L[:, :, 0]

        real_LAB = np.concatenate((L,)*3, axis=-1)
        real_LAB[:, :, 1:] = ab_test[i]

        predicted_LAB = np.concatenate((L,)*3, axis=-1)
        predicted_LAB[:, :, 1:] = ab_model[i]

        real_RGB = lab2rgb(real_LAB)
        pred_RGB = lab2rgb(predicted_LAB)

        plt.imshow(grey_channel, cmap="gray")
        plt.show()
        plt.imshow(real_RGB)
        plt.show()
        plt.imshow(pred_RGB)
        plt.show()

        # Print real LAB
        # Print predicted LAB


def predict_norm(model, datasets):
    l_test = datasets.test_L[:5]
    ab_test = datasets.test_ab[:5]

    ab_model = model.predict(l_test) * 128

    for i in range(5):
        # Print just L
        L = l_test[i]
        grey_channel = L[:, :, 0]

        real_LAB = np.concatenate((L,)*3, axis=-1)
        real_LAB[:, :, 1:] = ab_test[i]

        predicted_LAB = np.concatenate((L,)*3, axis=-1)
        predicted_LAB[:, :, 1:] = ab_model[i]

        real_RGB = lab2rgb(real_LAB)
        pred_RGB = lab2rgb(predicted_LAB)

        plt.imshow(grey_channel, cmap="gray")
        plt.show()
        plt.imshow(real_RGB)
        plt.show()
        plt.imshow(pred_RGB)
        plt.show()

        # Print real LAB
        # Print predicted LAB


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    datasets = Datasets()
    print("Completed loading datasets")

    if ARGS.model == 'cnn':
        model = CNNModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    elif ARGS.model == 'cnn-pre':
        # pre-trained CNN
        model = RESCNNModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 1)))
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        # Print summary of model
        model.summary()
    elif ARGS.model == 'gan':
        model = GANModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))

        # Print summary of model
        model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        if ARGS.model == 'cnn':
            model.load_weights(ARGS.load_checkpoint, by_name=False)
        # TODO: Add elif cases for other models
        elif ARGS.model == 'cnn-pre':
            model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # TODO add options for non-CNN models
    model.compile(
        optimizer=model.optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mean_squared_error", thresholded_loss])

    if ARGS.evaluate:
        predict_norm(model, datasets)
        test(model, datasets)
    else:
        train(model, datasets, checkpoint_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()


# How to show images
# def main():
#     datasets = Datasets()
#     color, gray = datasets.load_data()
#     color1 = color[0]
#     gray1 = gray[0]
#     plt.imshow(color1)
#     plt.show()
#     plt.imshow(gray1, cmap="gray")
#     plt.show()

#     print(color.shape, gray.shape)
#     return
