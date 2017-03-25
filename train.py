# cifar10

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/train', \
							""" Directory where to write event logs and checkpoints. """)
tf.app.flags.DEFINE_integer('max_step', 10000, \
							''' Number of batches to run. ''')
tf.app.flags.DEFINE_boolean('log_device_placement', False, \
							''' Whether to log device placement. ''')

def train():
	""" Train CIFAR-10 for a number of steps. """
	with tf.Graph().as_default():

		images, labels = cifar10.distorted_inputs()

def main(argv = None):
	cifar10.maybe_download_and_extract()
	if tf.gfile.Exists(FLAGS.train_dir):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)
	train()

if __name__ == '__main__':
	tf.app.run()