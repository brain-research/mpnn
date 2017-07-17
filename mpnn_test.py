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

# Simple unit tests for MPNNs.

import numpy as np
import tensorflow as tf
import mpnn


def build_feed_dict(ph, h, adjacency, dist, m):
  return {ph[0]: h, ph[1]: adjacency, ph[2]: dist, ph[3]: m}


def get_permutation_test_outputs(hparams):
  num_nodes = 4
  batch_size = 3
  input_dim = 5
  output_dim = 2

  with tf.Graph().as_default():
    model = mpnn.MPNN(hparams, input_dim, output_dim, num_edge_class=5)
    ph, _ = model.get_fprop_placeholders()
    pred_op = model.fprop(*ph)

    adjacency = np.random.randint(2, size=(batch_size, num_nodes, num_nodes))
    dist = np.random.rand(batch_size, num_nodes, num_nodes)
    h = np.random.rand(batch_size, num_nodes, input_dim)

    perm = np.random.permutation(num_nodes)

    h_perm = np.zeros_like(h)
    adjacency_perm = np.zeros_like(adjacency)
    dist_perm = np.zeros_like(dist)
    m = np.full((batch_size, num_nodes), 1)

    for i in xrange(len(h_perm)):
      h_perm[i] = h[i][perm]
    for i in xrange(len(adjacency_perm)):
      adjacency_perm[i] = adjacency[i][perm]
      dist_perm[i] = dist[i][perm]
      for j in xrange(len(adjacency_perm[i])):
        adjacency_perm[i][j] = adjacency_perm[i][j][perm]
        dist_perm[i][j] = dist_perm[i][j][perm]

    print h.shape, h_perm.shape
    print adjacency.shape, adjacency_perm.shape

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(
          pred_op, feed_dict=build_feed_dict(ph, h, adjacency, dist, m))
      output_perm = sess.run(
          pred_op,
          feed_dict=build_feed_dict(ph, h_perm, adjacency_perm, dist_perm, m))
      print "output no perm:"
      print output
      print "\noutput perm:"
      print output_perm
      return output, output_perm


def get_pad_test_outputs(hparams):
  # TODO(gilmer) This should test different paddings within the same batch,
  # in a similar way as in set2vec_test.py
  hparams = mpnn.MPNN.default_hparams()
  num_nodes = 4
  batch_size = 3
  input_dim = 5
  output_dim = 2
  pad = 3

  with tf.Graph().as_default():

    model = mpnn.MPNN(hparams, input_dim, output_dim, num_edge_class=5)

    ph, _ = model.get_fprop_placeholders()
    pred_op = model.fprop(*ph)

    adjacency = np.random.randint(2, size=(batch_size, num_nodes, num_nodes))
    dist = np.random.rand(batch_size, num_nodes, num_nodes)
    h = np.random.rand(batch_size, num_nodes, input_dim)
    m = np.full((batch_size, num_nodes), 1.0)
    h_pad = np.zeros((h.shape[0], h.shape[1] + pad, h.shape[2]))
    adjacency_pad = np.zeros((adjacency.shape[0], adjacency.shape[1] + pad,
                              adjacency.shape[2] + pad))

    dist_pad = np.zeros((dist.shape[0], dist.shape[1] + pad,
                         dist.shape[2] + pad))
    m_pad = np.zeros((batch_size, num_nodes + pad))

    for i in xrange(batch_size):
      for j in xrange(num_nodes):
        m_pad[i][j] = 1
    for i in xrange(len(h)):
      for j in xrange(len(h[i])):
        for k in xrange(len(h[i][j])):
          h_pad[i][j][k] = h[i][j][k]

    for i in xrange(len(adjacency)):
      for j in xrange(len(adjacency[i])):
        for k in xrange(len(adjacency[i][j])):
          adjacency_pad[i][j][k] = adjacency[i][j][k]
          dist_pad[i][j][k] = dist[i][j][k]

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(
          pred_op, feed_dict=build_feed_dict(ph, h, adjacency, dist, m))
      output_pad = sess.run(
          pred_op,
          feed_dict=build_feed_dict(ph, h_pad, adjacency_pad, dist_pad, m_pad))

      print "output no pad:"
      print output
      print "\noutput pad:"
      print output_pad

      return output, output_pad


class MPNNTest(tf.test.TestCase):
  """Tests for MPNNs."""

  def test_build_two_graphs(self):
    """Test constructing the MPNN graph."""
    batch_size = 5
    num_nodes = 3
    input_dim = 4
    output_dim = 6
    adjacency = np.random.randint(2, size=(batch_size, num_nodes, num_nodes))
    h = np.random.rand(batch_size, num_nodes, input_dim)
    dist = np.random.rand(batch_size, num_nodes, num_nodes)
    m = np.full((batch_size, num_nodes), 1)
    with tf.Graph().as_default():

      hparams = mpnn.MPNN.default_hparams()
      model = mpnn.MPNN(hparams, input_dim, output_dim, num_edge_class=5)
      ph, _ = model.get_fprop_placeholders()
      ph2, _ = model.get_fprop_placeholders()

      print ph[0], ph2[0]

      pred = model.fprop(*ph, train=True)
      pred2 = model.fprop(*ph2)

      with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        pred1 = sess.run(
            pred, feed_dict=build_feed_dict(ph, h, adjacency, dist, m))
        pred2 = sess.run(
            pred2, feed_dict=build_feed_dict(ph2, h, adjacency, dist, m))

    self.assertListEqual(list(pred1.shape), [batch_size, output_dim])
    self.assertListEqual(list(pred2.shape), [batch_size, output_dim])
    self.assertAllClose(pred1, pred2)
    print "Successfully constructed MPNN graph."

  def test_permutation_and_pad_invariance(self):
    # test GG-NN msg pass + graph level output
    hparams = mpnn.MPNN.default_hparams()
    output, output_perm = get_permutation_test_outputs(hparams)
    self.assertAllClose(output, output_perm)

    output, output_pad = get_pad_test_outputs(hparams)
    self.assertAllClose(output, output_pad)

    # test edge_network message function
    hparams.message_function = "edge_network"
    output, output_perm = get_permutation_test_outputs(hparams)
    self.assertAllClose(output, output_perm)

    output, output_pad = get_pad_test_outputs(hparams)
    self.assertAllClose(output, output_pad)

    # test edge_network + set2vec output
    hparams.output_function = "set2vec"
    output, output_perm = get_permutation_test_outputs(hparams)
    self.assertAllClose(output, output_perm)

    output, output_pad = get_pad_test_outputs(hparams)
    self.assertAllClose(output, output_pad)


if __name__ == "__main__":
  tf.test.main()
