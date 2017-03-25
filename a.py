# cifar10

from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/Users/wjh720/Desktop/now/SRT/Wavenet/Code/cifar10/model',
							''' Directory where to write event logs and checkpoints. ''')
tf.app.flags.DEFINE_integer('max_step', 10000,
							''' Number of batches to run. ''')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
							''' Whether to log device placement. ''')

def train():
	""" Train CIFAR-10 for a number of steps. """
	with tf.Graph().as_default():

		images, labels = cifar10.distorted_inputs()
