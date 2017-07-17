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
import graph_util
import set2vec


class Model(object):
  """Super class which uses tf.make_template."""

  def __init__(self, hparams):
    """Constructor.

    Args:
      hparams: tf.HParams object.
    """
    self.hparams = hparams

  def _fprop(self, *args, **kwargs):  # pylint: disable=unused-argument
    raise NotImplementedError("Model subclasses must define _fprop() method")

  def get_fprop_placeholders(self):
    """Returns the tuple *args,**kwargs used by the _fprop method."""
    raise NotImplementedError(
        "Model subclasses must define get_fprop_placeholders() method")

  # TODO(gilmer): This can be the Model's init function, and any subclass can
  # just call super().__init__()
  def init_fprop(self):
    """Initializes self.fprop. This should be called from subclasses' ctors.

    This function will contruct all of the variables defined with
    tf.get_variable in the sub classes fprop method and make a template
    out of the fprop method. In this way, instead of using variable scopes
    for variable reuse, the instantiation of the subclass will construct all
    of the model variables, and subsequent calls of the objects fprop method
    will add the fprop ops to the tensorflow graph using the tf variables
    which were defined when init_fprop was first called. This way variable
    reuse is trival by simply called model.fprop on different tensors. If you
    don't want to reuse variables, you will instead define a different model
    object.
    """
    scope_name = self.__class__.__name__

    self.fprop = tf.make_template(
        scope_name, self._fprop, create_scope_now_=True)

    if getattr(self.hparams, "use_placeholders", True):
      # Call self.fprop() to initialize variables in a dummy name scope
      # to manage the pollution
      with tf.name_scope("UNUSED"):
        args, kwargs = self.get_fprop_placeholders()
        self.fprop(*args, **kwargs)

  @property
  def params(self):
    """Returns string -> TFVariable mapping."""
    model_vars = self.fprop.variable_scope.trainable_variables()
    prefix = self.fprop.variable_scope.name + "/"
    basename_to_var = {}
    for v in model_vars:
      assert v.name.startswith(prefix)
      # Remove the :0 suffix and the scope prefix
      basename = v.name[len(prefix):-2]
      basename_to_var[basename] = v
    return basename_to_var


### Message Functions ###
# Every MPNN uses a message function M_t(h^t_v, h^t_w, e_vw) which computes
# a message vector for all ordered pairs of nodes. The aggregate incoming
# message to a node v at time step t is then
# a_v^t = sum M_t(h_v^t, h_w^t, e_{vw}).
#
# For some MPNNs we use the same message function at every timestep. In these
# cases there may be some common tensors that are computed the first time
# the fprop method is called, and then reused in subsequent calls of fprop.
# As a result, every message function class has an fprop API of the form:
# fprop(node_states, adjacency_in, reuse_graph_tensors = False),
# where reuse_graph_tensors is a boolean to indicate whether or it's the
# first time in the message passing phase that this is called.
def message_pass(node_states, a_in, a_out, node_dim):
  """Computes a_t from h_{t-1}, see bottom of page 3 in the paper.

  Args:
    node_states: [batch_size, num_nodes, node_dim] tensor (h_{t-1})
    a_in (tf.float32): [batch_size, num_nodes *node_dim, num_nodes * node_dim]
    a_out (tf.float32): [batch_size, num_nodes *node_dim, num_nodes * node_dim]
    node_dim (int32): dimension of node_states.

  Returns:
    messages (tf.float32): [batch_size, num_nodes, 2 * node_dim] For each pair
      of nodes in the graph a message is sent along both the incoming and
      outgoing edge.
  """
  batch_size = tf.shape(node_states)[0]
  num_nodes = tf.shape(node_states)[1]

  message_bias = tf.get_variable("message_bias", shape=2 * node_dim)
  h_flat = tf.reshape(
      node_states, [batch_size, num_nodes * node_dim, 1], name="h_flat")

  a_in_mul = tf.reshape(
      tf.matmul(a_in, h_flat), [batch_size * num_nodes, node_dim],
      name="a_in_mul")
  a_out_mul = tf.reshape(
      tf.matmul(a_out, h_flat), [batch_size * num_nodes, node_dim],
      name="a_out_mul")
  a_temp = tf.concat(
      [a_in_mul, a_out_mul], axis=1,
      name="a_temp")  # shape [batch_size * num_nodes, 2*node_dim]
  a_t = a_temp + message_bias
  messages = tf.reshape(a_t, [batch_size, num_nodes, 2 * node_dim])
  return messages


class GGNNMsgPass(Model):
  """Message passing function used in GG-NN.

  Embeds each edge type as a matrix A_e
  and the message from v -> w is A_e * h_v.
  """

  def __init__(self, hparams):
    """Build all of the variables.

    Args:
      hparams: tf.HParams object, only node_dim is relevant to this function.
    """
    super(GGNNMsgPass, self).__init__(hparams)
    self.node_dim = hparams.node_dim
    # TODO(gilmer): sub class should just call
    # super(self, GGNNMsgPass).__init__()
    # NOTE: init_fprop will set two member variables of the class, a_in
    # and a_out, these will be overwritten the first time fprop is called.
    self.init_fprop()

  def get_fprop_placeholders(self):
    return [
        tf.placeholder(tf.float32, shape=[None, None, self.hparams.node_dim]),
        tf.placeholder(
            tf.int32, shape=[None, None, None],
            name="adjacency_in_placeholder"),
        tf.placeholder(tf.float32, shape=[None, None, None],
                       name="distance")  # note this will be ignored
    ], {}

  def _precompute_graph(self, adjacency_in):
    """Precompute the a_in and a_out tensors.

    The a_in and a_out tensors are reused everytime frop is called, so we
    precompute them the first time fprop is called and then save the tensors
    as member variables.
    Args:
      adjacency_in: placeholder of integers of shape (batch_size, num_nodes,
                                                      num_nodes)
    """
    num_nodes = tf.shape(adjacency_in)[1]
    matrices_in = tf.get_variable(
        "adjacency_weights_in",
        shape=[self.hparams.num_edge_class, self.node_dim, self.node_dim])

    matrices_out = tf.get_variable(
        "adjacency_weights_out",
        shape=[self.hparams.num_edge_class, self.node_dim, self.node_dim])

    zeros = tf.constant(0.0, shape=[1, self.node_dim, self.node_dim])

    # these are the matrices corresponding to the incoming edge labels
    # edge label 0 corresponds to a non-edge
    if self.hparams.non_edge:
      non_edge_in = tf.get_variable(
          "non_edge_in", shape=[1, self.node_dim, self.node_dim])

      matrices_in = tf.concat(
          [non_edge_in, matrices_in], axis=0, name="matrices_in")
    else:
      matrices_in = tf.concat([zeros, matrices_in], axis=0, name="matrices_in")

    # matrices corresponding to the outgoing edge labels

    if self.hparams.non_edge:
      non_edge_out = tf.get_variable(
          "non_edge_out", shape=[1, self.node_dim, self.node_dim])

      matrices_out = tf.concat(
          [non_edge_out, matrices_out], axis=0, name="matrices_out")
    else:
      matrices_out = tf.concat(
          [zeros, matrices_out], axis=0, name="matrices_out")

    adjacency_out = tf.transpose(adjacency_in, [0, 2, 1])
    a_in = tf.gather(matrices_in, adjacency_in)
    a_out = tf.gather(matrices_out, adjacency_out)

    # Make a_in and a_out have shape [batch_size, n*d, n*d]
    # the node repsentations are shape [batch_size, n*d] so we can use
    # tf.matmul with a_in and the node vector
    # The transpose is neccessary to make the reshape read the elements of A
    # in the correct order (reshape reads lexicographically starting with
    # index [0][0][0][0] -> [0][0][0][1] -> ...)
    a_in = tf.transpose(a_in, [0, 1, 3, 2, 4])
    self._a_in = tf.reshape(
        a_in, [-1, num_nodes * self.node_dim, num_nodes * self.node_dim])
    a_out = tf.transpose(a_out, [0, 1, 3, 2, 4])
    self._a_out = tf.reshape(
        a_out, [-1, num_nodes * self.node_dim, num_nodes * self.node_dim])

  def _fprop(
      self,
      node_states,
      adjacency_in,
      distance,  # pylint: disable=unused-argument
      reuse_graph_tensors=False):
    """Computes a_t from h_{t-1}, see bottom of page 3 in the paper.

    Args:
      node_states: [batch_size, num_nodes, node_dim] tensor (h_{t-1})
      adjacency_in (tf.int32): [batch_size, num_nodes, num_nodes]
      distance (tf.float): [batch_size, num_nodes, num_nodes] NOT USED.
      reuse_graph_tensors: (boolean) must be set to True the first time that
        fprop is called so that we can compute the a_in and a_out tensors.

    Returns:
     a_t: [batch_size * num_nodes, node_dim] which is the node represenations
     after a single propgation step

     This also sets graph_precomputed to True to indicate that part of the
     graph has been cached and will be reused in future calls of _fprop
    """
    # build the larger A matrices on the first call of _fprop
    if not reuse_graph_tensors:
      self._precompute_graph(adjacency_in)

    return message_pass(node_states, self._a_in, self._a_out,
                        self.hparams.node_dim)


class EdgeNetwork(Model):
  """Message passing function used in GG-NN.

  A feed forward neural network is applied to each edge in the adjacency matrix,
  which is assumed to be vector valued. It maps the edge vector to a
  node_dim x node_dim matrix, denoted NN(e). The message from node v -> w is
  then NN(e) h_v. This is a generalization of the message function in the
  GG-NN paper, which embeds the discrete edge label as a matrix.
  """

  def __init__(self, hparams):
    """Build all of the variables.

    Args:
      hparams: tf.HParams object, only node_dim is relevant to this function.
    """
    super(EdgeNetwork, self).__init__(hparams)
    self.init_fprop()

  def get_fprop_placeholders(self):
    node_ph = tf.placeholder(
        tf.float32, shape=[None, None, self.hparams.node_dim], name="node_ph")
    adjacency_ph = tf.placeholder(
        tf.int32, shape=[None, None, None], name="adjacency_in_ph")
    distance = tf.placeholder(
        tf.float32, shape=[None, None, None], name="distance")
    return [node_ph, adjacency_ph, distance], {}

  def _precompute_graph(self, adjacency_in, distance):
    """Precompute the a_in and a_out tensors.

    (we don't want to add to the graph everytime _fprop is called)
    Args:
      adjacency_in: placeholder of integers of shape (batch_size, num_nodes,
                                                      num_nodes)
      distance: placeholder of floats of shape (batch_size,
              num_nodes, num_nodes)
    """
    batch_size = tf.shape(adjacency_in)[0]
    num_nodes = tf.shape(adjacency_in)[1]

    distance_exp = tf.expand_dims(distance, 3)

    adjacency_in_one_hot = tf.one_hot(
        adjacency_in, self.hparams.num_edge_class, name="adjacency_in_one_hot")
    adjacency_in_w_dist = tf.concat(
        [adjacency_in_one_hot, distance_exp],
        axis=3,
        name="adjacency_in_w_dist")  # [batch_size, num_nodes, num_nodes
    edge_dim = adjacency_in_w_dist.get_shape()[3].value

    # build the edge_network for incoming edges
    with tf.name_scope("adj_in_edge_nn"):
      with tf.variable_scope("adj_in_edge_nn"):
        adj_reshape_in = tf.reshape(
            adjacency_in_w_dist, [batch_size * num_nodes * num_nodes, edge_dim],
            name="adj_reshape_in")
        a_in_tmp = graph_util.feed_forward_nn(
            adj_reshape_in,
            self.hparams.edge_num_layers,
            self.hparams.node_dim**2,
            hidden_dim=self.hparams.edge_hidden_dim,
            activation=self.hparams.activation,
            normalizer=self.hparams.normalizer)
      a_in_tmp = tf.reshape(a_in_tmp, [
          batch_size, num_nodes, num_nodes, self.hparams.node_dim,
          self.hparams.node_dim
      ])

      a_in = tf.reshape(
          tf.transpose(a_in_tmp, [0, 1, 3, 2, 4]), [
              -1, num_nodes * self.hparams.node_dim,
              num_nodes * self.hparams.node_dim
          ],
          name="a_in")

    adjacency_out = tf.transpose(adjacency_in, [0, 2, 1])
    adjacency_out_one_hot = tf.one_hot(
        adjacency_out,
        self.hparams.num_edge_class,
        name="adjacency_out_one_hot")
    adjacency_out_w_dist = tf.concat(
        [adjacency_out_one_hot, distance_exp],
        axis=3,
        name="adjacency_out_w_dist")  # [batch_size, num_nodes, num_nodes
    with tf.name_scope("adj_out_edge_nn"):
      with tf.variable_scope("adj_out_edge_nn"):
        adj_reshape_out = tf.reshape(
            adjacency_out_w_dist,
            [batch_size * num_nodes * num_nodes, edge_dim],
            name="adj_reshape_out")
        a_out_tmp = graph_util.feed_forward_nn(
            adj_reshape_out,
            self.hparams.edge_num_layers,
            self.hparams.node_dim**2,
            hidden_dim=self.hparams.edge_hidden_dim,
            activation=self.hparams.activation,
            normalizer=self.hparams.normalizer)
      a_out_tmp = tf.reshape(a_out_tmp, [
          batch_size, num_nodes, num_nodes, self.hparams.node_dim,
          self.hparams.node_dim
      ])
      a_out = tf.reshape(
          tf.transpose(a_out_tmp, [0, 1, 3, 2, 4]), [
              -1, num_nodes * self.hparams.node_dim,
              num_nodes * self.hparams.node_dim
          ],
          name="a_out")

    self._a_in = a_in
    self._a_out = a_out

  def _fprop(self,
             node_states,
             adjacency_in,
             distance,
             reuse_graph_tensors=False):
    """Computes a_t from h_{t-1}, see bottom of page 3 in the paper.

    Args:
      node_states: [batch_size, num_nodes, node_dim] tensor (h_{t-1})
      adjacency_in: [batch_size, num_nodes, num_nodes] (tf.int32)
      distance: [batch_size, num_nodes, num_nodes] (tf.float32)
      reuse_graph_tensors: Boolean to indicate whether or not the self._a_in
         should be reused or not. Should be set to False on first call, and True
         on subsequent calls.

    Returns:
     a_t: [batch_size * num_nodes, node_dim] which is the node represenations
     after a single propgation step

     This also sets graph_precomputed to True to indicate that part of the
     graph has been cached and will be reused in future calls of _fprop
    """
    if not reuse_graph_tensors:
      self._precompute_graph(adjacency_in, distance)

    return message_pass(node_states, self._a_in, self._a_out,
                        self.hparams.node_dim)


### UPDATE FUNCTIONS ###
# Every MPNN uses an update function to update the vertex hidden states at
# each time step. The update equation is of the form
# h_v^{t+1} = U_t(h_v^t, m_v^{t+1}. Where m_v^{t+1} is the aggregate message
# vector computed using the message function at time step t.
class GRUUpdate(Model):
  """Update function used in GG-NN."""

  def __init__(self, hparams):
    """GRU update function used in GG-NN.

    Implements h_v^{t+1} = GRU(h_v^t, m_v^{t+1}).

    Args:
      hparams (tf.HParams object): only relevant hparam is node_dim which is the
        dimension of the node states.
    """
    super(GRUUpdate, self).__init__(hparams)
    self.node_dim = hparams.node_dim

    self.init_fprop()

  def get_fprop_placeholders(self):
    node_input = tf.placeholder(
        tf.float32, shape=[None, None, self.node_dim], name="node_input")
    messages = tf.placeholder(
        tf.float32,
        shape=[None, None, 2 * self.node_dim],
        name="messages_input")
    mask = tf.placeholder(tf.bool, shape=[None, None], name="mask")
    return [node_input, messages, mask], {}

  def _fprop(self, node_states, messages, mask):
    """Build the fprop graph.

    Args:
      node_states: [batch_size, num_nodes, node_dim] tensor (h_{t-1})
      messages: [batch_size, num_nodes, 2*node_dim] (a_t from the GGNN paper)
      mask: [batch_size, num_nodes], 0 if this node doesn't exist
        (padded), 1 otherwise (tf.bool)

    Returns:
      updated_states: [batch_size, num_nodes, node_dim]
    """

    batch_size = tf.shape(node_states)[0]
    num_nodes = tf.shape(node_states)[1]
    mask_col = tf.cast(
        tf.reshape(mask, [batch_size * num_nodes, 1]),
        tf.float32,
        name="mask_col")

    w_z = tf.get_variable("GRU_w_z", shape=[2 * self.node_dim, self.node_dim])
    u_z = tf.get_variable("GRU_u_z", shape=[self.node_dim, self.node_dim])
    w_r = tf.get_variable("GRU_w_r", shape=[2 * self.node_dim, self.node_dim])
    u_r = tf.get_variable("GRU_u_r", shape=[self.node_dim, self.node_dim])
    w = tf.get_variable("GRU_w", shape=[2 * self.node_dim, self.node_dim])
    u = tf.get_variable("GRU_u", shape=[self.node_dim, self.node_dim])

    nodes_rs = tf.reshape(
        node_states, [batch_size * num_nodes, self.node_dim], name="nodes_rs")
    messages_rs = tf.reshape(
        messages, [batch_size * num_nodes, 2 * self.node_dim],
        name="messages_rs")

    z_t = tf.sigmoid(
        tf.matmul(messages_rs, w_z) + tf.matmul(nodes_rs, u_z), name="z_t")
    r_t = tf.sigmoid(
        tf.matmul(messages_rs, w_r) + tf.matmul(nodes_rs, u_r), name="r_t")

    # tanh ( w a_v^t + u (r_v^t dot h_v^(t-1)))
    h_tilde = tf.tanh(
        tf.matmul(messages_rs, w) + tf.matmul(tf.multiply(r_t, nodes_rs), u),
        name="h_tilde")

    # h_t has shape [batch_size * num_nodes, node_dim]
    h_t = tf.multiply(1 - z_t, nodes_rs) + tf.multiply(z_t, h_tilde)
    h_t_masked = tf.multiply(
        h_t, mask_col, name="mul_h_t_masked"
    )  # zero out the non nodes (correct for the bias term

    h_t_rs = tf.reshape(
        h_t_masked, [batch_size, num_nodes, self.node_dim], name="h_t_rs")
    return h_t_rs


### READOUT FUNCTIONS ###
# Every MPNN uses a readout function to compute the final output of the network.
# Currently we only implement graph level readouts, which map from the final
# hidden states of all the nodes to a single vector valued target for the entire
# graph. That is \hat{y} = R(\{h_v^T : v \ in G}). The only requirement is that
# the readout is invariant to the order of nodes, if this is satisfied then the
# output of the MPNN will be invariant to the order of nodes.
#
# It is also interesting to consider cases where you have a target at each
# vertex in the graph. In this case, the message passing phase does not need
# to change at all, instead you can just apply a readout function to each
# final node state independantly. E.g. \hat{y}_v = R(h_v^T), for all v.
#
# One can also consider edge level targets, say for example we want to predict
# the spatial distance between pairs of nodes. We have not tested this
# extensively, we did try feeding in the pair of final node states to a readout,
# e.g. \hat{y}_{vw} = R(h_v^T, h_w^T), however this did not work very well.
# This might be a setting where having edge states that evolve during message
# passing will work better, because then we can just readout from the final
# edge state, e.g. \hat{y}_{vw} = R(e_{vw}^T).


class GraphLevelOutput(Model):
  """Graph Level Output from the GG-NN paper."""

  def __init__(self, input_dim, output_dim, hparams):
    """Class used to map final node states to an output vector.

    Args:

      input_dim: Dimension of the node states taken as input
      output_dim: Dimension of the vector valued output of the network
      hparams: Specifies the architecture of the output neural nets.

    Relevant hparams for this function:
      hparams.num_output_hidden_layers: (int) number of hidden layers in the
        output
      neural nets
      hparams.hidden_dim: (int) hidden dim shared by all hidden layers.
      hparams.activation: (str - 'relu' or 'tanh') indicates what activation fct
      to use in the neural nets
      hparams.normalizer: (str - 'layer' or 'none') whether or not to use layer
      norm in the neural nets
      hparams.keep_prob: (float) dropout keep prob for the output neural nets

    """
    super(GraphLevelOutput, self).__init__(hparams)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.init_fprop()

  def get_fprop_placeholders(self):
    node_states = tf.placeholder(
        tf.float32, shape=[None, None, self.input_dim], name="node_states")
    mask = tf.placeholder(tf.bool, shape=[None, None], name="mask")
    return [node_states, mask], {}

  def _fprop(self, node_states, mask, train=False):
    """Creates the output network for the entire graph.

    Args:
      node_states: shape = [batch_size, num_nodes, input_dim]
      mask: shape = [batch_size, num_nodes], representing which nodes
        appear in the graph (tf.bool)
      train (Boolean): Indicates whether or not the graph is for training.
    Returns:
      tensor with shape = [batch_size, output_dim] representing the final output
      of the network as in equation (7) on page 4 of the ggnn paper.
    """
    if train:
      keep_prob = self.hparams.keep_prob
    else:
      keep_prob = 1.0

    batch_size = tf.shape(node_states)[0]
    num_nodes = tf.shape(node_states)[1]
    mask_col = tf.cast(
        tf.reshape(mask, [batch_size * num_nodes, 1]),
        tf.float32,
        name="mask_col")
    nodes_col = tf.reshape(
        node_states, [batch_size * num_nodes, self.input_dim], name="nodes_col")

    with tf.variable_scope("i_feedforward"):
      i_output = graph_util.feed_forward_nn(
          nodes_col,
          self.hparams.num_output_hidden_layers,
          self.output_dim,
          keep_prob,
          self.hparams.hidden_dim,
          activation=self.hparams.activation,
          normalizer=self.hparams.normalizer)
    with tf.variable_scope("j_feedforward"):
      j_output = graph_util.feed_forward_nn(
          nodes_col,
          self.hparams.num_output_hidden_layers,
          self.output_dim,
          keep_prob,
          self.hparams.hidden_dim,
          activation=self.hparams.activation,
          normalizer=self.hparams.normalizer)

    # NOTE: YuJia's paper has tanh here, but mentions that identity will work
    # For regression tasks identity is VERY important here.
    # gated_activations = tf.multiply(tf.sigmoid(i_output), tf.tanh(j_output))
    gated_activations = tf.multiply(tf.sigmoid(i_output), j_output)
    # gated_activations has shape [batch_size * num_nodes, output_dim]

    gated_activations = tf.multiply(gated_activations,
                                    mask_col)  # dropout the non nodes again

    gated_activations = tf.reshape(gated_activations,
                                   [batch_size, num_nodes, self.output_dim])

    # sum over all nodes in the graphs
    # when we define the loss we will do a softmax over this vector
    final_output = tf.reduce_sum(gated_activations, 1)

    return final_output


class Set2VecOutput(Model):
  """Set2Set output Output from Vinyals et. al."""

  def __init__(self, input_dim, output_dim, hparams):
    """Class used to map final node states to an output vector.

    The output of the fprop method will be a target vector of dimension
    output_dim which will be invariant to the order of elements in the
    "node_states" tensor. This implements the "process" block described in
    https://arxiv.org/pdf/1511.06391.pdf. For more detailed documentation,
    see set2vec.py.

    Args:

      input_dim: Dimension of the node states taken as input
      output_dim: Dimension of the vector valued output of the network
      hparams: Specifies the architecture of the output neural nets.

    Relevant hparams for this function:
      hparams.num_output_hidden_layers: (int) number of hidden layers in the
        output
      neural nets
      hparams.hidden_dim: (int) hidden dim shared by all hidden layers.
      hparams.activation: (str - 'relu' or 'tanh') indicates what activation fct
      to use in the neural nets
      hparams.normalizer: (str - 'layer' or 'none') whether or not to use layer
      norm in the neural nets
      hparams.keep_prob: (float) dropout keep prob for the output neural nets

    """
    super(Set2VecOutput, self).__init__(hparams)
    self.hparams = hparams
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.init_fprop()

  def get_fprop_placeholders(self):
    node_states = tf.placeholder(
        tf.float32, shape=[None, None, self.input_dim], name="node_states")
    mask = tf.placeholder(tf.bool, shape=[None, None], name="mask")
    return [node_states, mask], {}

  def _fprop(self, node_states, mask, train=False):
    """Creates the output network for the entire graph.

    Args:
      node_states: shape = [batch_size, num_nodes, input_dim]
      mask: (tf.bool, shape = [batch_size, num_nodes]), representing which nodes
        appear in the graph
      train (Boolean): Indicates whether or not the graph is for training.
    Returns:
      tensor with shape = [batch_size, output_dim] representing the final output
      of the network as in equation (7) on page 4 of the ggnn paper.
    """
    if train:
      keep_prob = self.hparams.keep_prob
    else:
      keep_prob = 1.0

    node_dim = int(node_states.shape[2])

    nodes_exp = tf.expand_dims(node_states, 2, "nodes_exp")
    attention_w1 = tf.get_variable("att_W_1", [1, 1, node_dim, node_dim])
    embedded_nodes = tf.nn.conv2d(
        nodes_exp, attention_w1, [1, 1, 1, 1], "SAME", name="embedded_nodes"
    )  # embed the elements of the final set of outputs.

    _, _, m = set2vec.set2vec(
        embedded_nodes,
        self.hparams.set2set_comps,
        mask=mask,
        inner_prod=self.hparams.inner_prod,
        name="output_LSTMLoopAtt")
    # m has shape [batch_size, 2* node_dim]

    with tf.name_scope("FeedForwardNN"):
      output = graph_util.feed_forward_nn(
          m,
          self.hparams.num_output_hidden_layers,
          self.output_dim,
          keep_prob,
          self.hparams.hidden_dim,
          activation=self.hparams.activation,
          normalizer=self.hparams.normalizer)
    return output


class MPNN(Model):
  r"""Class which implements a general MPNN.

  See https://arxiv.org/abs/1704.01212 for background on MPNNs.
  Every MPNN is defined in terms of message functions, vertex update functions
  and a readout function.

  Message Functions: These are used to compute messages between all pairs of
  connected nodes in the graph. The message from w -> v at time step t+1 is
  m_vw^{t+1} M_t(h^t_v, h^t_w, e_vw). The aggregate incoming message to vertex
  v at time step t+1 is then m_v^{t+1} = sum_w m_vw^{t+1}. A message function
  class computes the aggregate message vectors m_v^{t+1} for all nodes in
  parallel. Thus their fprop API is:
  fprop(node_states, adjacency_in, reuse_graph_tensors)
  and they return a tensor of shape (batch_size, num_nodes, message_dim) which
  are the aggregate message vectors m_v^{t+1}.

  Update Functions: These are used to update the hidden states at each vertex
  as a function of the previous hidden state, and the incoming message at that
  time step. h_v^{t+1} = U_t(h_v^{t}, m_v^{t+1}). The API is then
  fprop(node_states, messages, mask) where messages is returned from the fprop
  of a message class. The mask is neccessary in order to allow batches to
  contain graphs of different sizes. The mask indicates which nodes
  in each graph are actually present.

  Readout Function: This computes the graph level output from the final node
  states after T steps of message passing have been performed.
  e.g. \hat{y} = R(\{h_v^T\}). The fprop api is then
  fprop(final_node_states, mask, train)
  where the final node states are whats output from the message passing phase,
  mask indicates which nodes are actually present in the graph, and train
  is a boolean to indicate if this is the train graph or not (relevant for
  the dropout keep_prob).
  """

  @staticmethod
  def default_hparams():
    return tf.contrib.training.HParams(
        set2set_comps=12,
        non_edge=0,
        node_dim=50,
        num_propagation_steps=6,
        num_output_hidden_layers=1,
        max_grad_norm=4.0,
        batch_size=20,
        optimizer="adam",
        momentum=.9,  # only used if optimizer is set to momentum
        init_learning_rate=.00013,
        decay_factor=.5,  # final learning rate will be initial*.1
        decay_every=500000,  # how often to decay the lr (#batches)
        reuse=True,  # use the same message and update weights at each time step
        message_function="matrix_multiply",
        update_function="GRU",
        output_function="graph_level",
        hidden_dim=200,
        keep_prob=1.0,  # in our experiments dropout did not help
        edge_num_layers=4,
        edge_hidden_dim=50,
        propagation_type="normal",
        activation="relu",
        normalizer="none",
        inner_prod="default"  #inner product similarity to use for set2vec
    )

  def __init__(self, hparams, input_dim, output_dim, num_edge_class):
    """Construct an MPNN.

    Args:
      hparams (tf.HParams object): See default hparams for all possibilities.
      input_dim: (int) dimension of the input node states
      output_dim: (int) dimension of the vector valued output
      num_edge_class: (int) number of edge types for the adjacency matrix

    Raises:
      Exception: If any invalid hparams are set.
    """
    super(MPNN, self).__init__(hparams)
    self.hparams.num_edge_class = num_edge_class

    self.input_dim = input_dim
    self.output_dim = output_dim

    if self.hparams.message_function == "matrix_multiply":
      message_class = GGNNMsgPass
    elif self.hparams.message_function == "edge_network":
      message_class = EdgeNetwork
    else:
      raise ValueError(
          "Invalid message function: {}".format(self.hparams.message_function))

    if self.hparams.update_function == "GRU":
      update_class = GRUUpdate
    else:
      raise ValueError(
          "Invalid update function: {}".format(self.hparams.update_function))

    if self.hparams.output_function == "graph_level":
      output_class = GraphLevelOutput
    elif self.hparams.output_function == "set2vec":
      output_class = Set2VecOutput
    else:
      raise ValueError(
          "Invalid output function {}".format(self.hparams.output_function))

    if self.hparams.reuse:
      self.message_functions = [message_class(self.hparams)]
      self.update_functions = [update_class(self.hparams)]
    else:
      self.message_functions = [
          message_class(self.hparams)
          for _ in xrange(self.hparams.num_propagation_steps)
      ]
      self.update_functions = [
          update_class(self.hparams)
          for _ in xrange(self.hparams.num_propagation_steps)
      ]

    self.output_function = output_class(input_dim + self.hparams.node_dim,
                                        output_dim, self.hparams)

    self.init_fprop()

  def get_fprop_placeholders(self):
    # TODO(gilmer): Rethink the placeholders for the MPNN class. Right now
    # we assume there is an integer valued adjacency_in input matrix, and
    # a separate distance matrix of floats. Some MPNN's need the integer
    # valued assumtion (vanilla GG-NN) but others it makes more sense
    # to just have a vector valued adjacency matrix.
    node_input = tf.placeholder(
        tf.float32, shape=[None, None, self.input_dim], name="node_input")
    adjacency_in = tf.placeholder(
        tf.int32, shape=[None, None, None], name="adjacency_in")
    distance = tf.placeholder(
        tf.float32, shape=[None, None, None], name="distance")
    mask = tf.placeholder(tf.bool, shape=[None, None], name="mask_input")
    return [node_input, adjacency_in, distance, mask], {}

  def _fprop(self, node_input, adjacency_in, distance, mask, train=False):
    """Builds the model graph.

    Args:
      node_input: placeholder of shape (batch_size, num_nodes, node_dim)
      adjacency_in: placeholder of shape (batch_size, num_nodes, num_nodes,
      edge_dim)
      distance: tf.float32 tensor of shape (batch_size, num_nodes, num_nodes),
          contains the distance matrix of the molecule.
      mask: placeholder of shape (batch_size, num_nodes), used when batches
         contain graphs of different sizes, 1 specifies existence of a node,
         0 specifies no node. tf.bool
      train: (Boolean) Is this graph for training (relevant for keep_prob)?
    Returns:
      final_output: tensor of shape [None, self.output_dim].
    """
    # node_dim is the internal hidden dim, we pad up to this dimension
    # shape = [None, None, node_dim]
    input_node_dim = tf.shape(node_input)[2]
    with tf.control_dependencies(
        [tf.assert_less_equal(input_node_dim, self.hparams.node_dim)]):
      padded_nodes = tf.pad(node_input, [
          [0, 0],
          [0, 0],
          [0, self.hparams.node_dim - input_node_dim],
      ])

    # this will be the initial node representation vector h^0, it is the
    # concatenation of all of the node representations
    # shape [batch_size * num_nodes, node_dim] will allow matrix
    # multiplication accross batches using right matrix mult by a
    # node_dim x node_dim matrix
    h_list = [padded_nodes]

    for i in xrange(self.hparams.num_propagation_steps):
      if self.hparams.reuse:
        messages = self.message_functions[0].fprop(
            h_list[-1], adjacency_in, distance, reuse_graph_tensors=(i != 0))
        new_h = self.update_functions[0].fprop(h_list[-1], messages, mask)
      else:
        messages = self.message_functions[i].fprop(h_list[-1], adjacency_in,
                                                   distance)
        new_h = self.update_functions[i].fprop(h_list[-1], messages, mask)
      h_list.append(new_h)
    h_concat_x = tf.concat([h_list[-1], node_input], axis=2, name="h_concat_x")

    final_output = self.output_function.fprop(h_concat_x, mask, train)

    return final_output
