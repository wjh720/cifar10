# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/Users/wjh720/Desktop/Tmp/cifar10/cifar10_train_1',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_checkpoints', 5,
                            """The number of checkpoints.""")
tf.app.flags.DEFINE_integer('checkpoint_every', 1000,
                            """Number of batches to make checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def save(saver, sess, logdir, step):
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
  print('Storing checkpoint to {} ...'.format(logdir), end="")
  sys.stdout.flush()

  if not os.path.exists(logdir):
      os.makedirs(logdir)

  saver.save(sess, checkpoint_path, global_step=step)
  print(' Done.')

def load(saver, sess, logdir):
  print("Trying to restore saved checkpoints from {} ...".format(logdir),
        end="")

  ckpt = tf.train.get_checkpoint_state(logdir)
  if ckpt:
      print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
      global_step = int(ckpt.model_checkpoint_path
                        .split('/')[-1]
                        .split('-')[-1])
      print("  Global step was: {}".format(global_step))
      print("  Restoring...", end="")
      saver.restore(sess, ckpt.model_checkpoint_path)
      print(" Done.")
      return global_step
  else:
      print(" No checkpoint found.")
      return None

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    #global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    optim = cifar10.train(loss)

    logdir = FLAGS.train_dir
    restore_from = logdir

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=FLAGS.max_checkpoints)

    try:
      saved_global_step = load(saver, sess, restore_from)
      if saved_global_step is None:
          # The first training step will be saved_global_step + 1,
          # therefore we put -1 here for new or overwritten trainings.
          saved_global_step = -1

    except:
      print("Something went wrong while restoring checkpoint. "
            "We will terminate training to avoid accidentally overwriting "
            "the previous model.")
      raise

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      if step % 100 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        summary, loss_value, _ = sess.run([summaries, loss, optim], options=run_options, 
                                          run_metadata=run_metadata)
        writer.add_summary(summary, step)
        writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
      else:
        summary, loss_value, _ = sess.run([summaries, loss, optim])
        writer.add_summary(summary, step)

      duration = time.time() - start_time
      print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

      if step % FLAGS.checkpoint_every == 0:
        save(saver, sess, logdir, step)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
