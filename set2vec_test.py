#
# Copyright 2009 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# Simple unit tests for set2vec.

# python set2vec_test.py

import numpy as np
import tensorflow as tf

import set2vec

tf.app.flags.DEFINE_integer("test_random_seed", 0, "random seed to use for the"
                            "tests")

FLAGS = tf.app.flags.FLAGS


class Set2VecTest(tf.test.TestCase):
  """Tests for Set2Vec."""

  def test_permutation_invariance(self):
    np.random.seed(seed=FLAGS.test_random_seed)
    num_nodes = 4
    batch_size = 3
    input_dim = 5
    num_timesteps = 10

    with tf.Graph().as_default():
      input_ph = tf.placeholder(tf.float32, [None, None, 1, input_dim])
      _, _, m = set2vec.set2vec(input_ph, num_timesteps)

      input_np = np.random.randn(batch_size, num_nodes, 1, input_dim)
      input_np_perm = input_np[:, np.random.permutation(num_nodes), :, :]

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(m, feed_dict={input_ph: input_np})
        out_perm = sess.run(m, feed_dict={input_ph: input_np_perm})

      self.assertAllClose(out, out_perm)

  def test_pad_invariance(self):
    np.random.seed(seed=FLAGS.test_random_seed)
    num_nodes = 4
    batch_size = 3
    input_dim = 5
    num_timesteps = 10
    pad = 2

    with tf.Graph().as_default():
      input_ph = tf.placeholder(tf.float32, [None, None, 1, input_dim])
      mask = tf.placeholder(tf.bool, [None, None])
      _, _, m = set2vec.set2vec(input_ph, num_timesteps, mask=mask)

      input_np = np.random.randn(batch_size, num_nodes, 1, input_dim)
      tmp_input_pad = np.ones((batch_size, num_nodes + pad, 1, input_dim))
      tmp_input_pad[:, :num_nodes, :, :] = input_np
      input_np_pad = tmp_input_pad[:]

      mask_np = np.ones((batch_size, num_nodes))
      tmp_mask_pad = np.zeros((batch_size, num_nodes + pad))
      tmp_mask_pad[:, :num_nodes] = mask_np
      mask_np_pad = tmp_mask_pad[:]

      # Permute the masks and inputs for each element in the batch.
      # We create separate permutation for each element in order to make the
      # test more general.
      for i in xrange(batch_size):
        perm = np.random.permutation(mask_np_pad.shape[1])
        mask_np_pad[i, :] = tmp_mask_pad[i, perm]
        input_np_pad[i, :] = tmp_input_pad[i, perm]

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(m, feed_dict={input_ph: input_np, mask: mask_np})
        out_pad = sess.run(
            m, feed_dict={input_ph: input_np_pad,
                          mask: mask_np_pad})

      self.assertAllClose(out, out_pad)


if __name__ == "__main__":
  tf.test.main()
