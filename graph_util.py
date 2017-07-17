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

import tensorflow as tf


def feed_forward_nn(input_tensor,
                    num_hidden_layers,
                    output_dim,
                    keep_prob=None,
                    hidden_dim=-1,
                    activation="tanh",
                    normalizer="none"):
  """Creates a fully connected feed forward neural network.

  Args:
    input_tensor: shape [batch_size*num_nodes, input_dim], assumed to be
       the final node states after the propgation step concat with the
       initial nodes.
    num_hidden_layers (int32): number of hidden layers in the network
        set to 0 for a linear network.
    output_dim (int32): dimension of the output of the network.
    keep_prob (scalar tensor or float): Dropout keep prob.
    hidden_dim (int32): size of the hidden layers
    activation (string): tanh or relu
    normalizer (string): layer or none

  Returns:
    tensor of shape [batch_size * num_nodes, output_dim]
    note there is no non-linearity applied to the output.

  Raises:
    Exception: If given activation or normalizer not supported.
  """
  if activation == "tanh":
    act = tf.tanh
  elif activation == "relu":
    act = tf.nn.relu
  else:
    raise ValueError("Invalid activation: {}".format(activation))

  if normalizer == "layer":
    norm = tf.contrib.layers.layer_norm
  elif normalizer == "none":
    norm = None
  else:
    raise ValueError("Invalid normalizer: {}".format(normalizer))

  h_nn = input_tensor  # first set of "hidden" units is the input

  for i in xrange(num_hidden_layers):
    with tf.name_scope("fully_connected/layer{}".format(i + 1)):
      layer_dim = h_nn.get_shape()[1].value
      w = tf.get_variable("W{}".format(i), shape=[layer_dim, hidden_dim])
      b = tf.get_variable("b{}".format(i), shape=[hidden_dim])
      h_nn = act(tf.matmul(h_nn, w) + b)

      if norm is not None:
        h_nn = norm(h_nn)
      if keep_prob is not None:
        h_nn = tf.nn.dropout(h_nn, keep_prob)

      tf.summary.histogram("h_nn{}".format(i), h_nn)

  layer_dim = h_nn.get_shape()[1].value
  output_w = tf.get_variable("output_W", shape=[layer_dim, output_dim])
  output_b = tf.get_variable("output_b", shape=[output_dim])

  # final output has no non-linearity, this is applied outside this function
  nn_output = tf.matmul(h_nn, output_w) + output_b
  return nn_output
