import tensorflow as tf

from tframe import checker
from tframe import pedia
from tframe import Classifier

from tframe.layers import Input, Linear, Activation
from tframe.layers import BatchNorm
from tframe.models.recurrent import Recurrent
from tframe.nets.rnn_cells.basic_cell import BasicRNNCell
from tframe.nets.rnn_cells.lstms import BasicLSTMCell
from tframe.nets.rnn_cells.gru import GRU
from tframe.nets.rnn_cells.amu import AMU
from tframe.models import Recurrent

from tframe.config import Config
import tframe.metrics as metrics


def lstm0(th):
  assert isinstance(th, Config)
  th.mark = 'lstm_' + th.mark
  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)

  # Add input layer
  model.add(Input(sample_shape=th.input_shape))

  # Add lstm cells
  for dim in th.rc_dims:
    model.add(BasicLSTMCell(state_size=dim))

  # Add output layer
  model.add(Linear(output_dim=th.output_dim))

  # Build model
  optimizer = tf.train.AdamOptimizer(th.learning_rate)
  model.build(optimizer)

  return model


def fc_lstm(th):
  assert isinstance(th, Config)
  th.mark = 'fc_lstm_' + th.mark
  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)

  # Add input layer
  model.add(Input(sample_shape=th.input_shape))

  # Add fc layers
  for dim in th.fc_dims:
    checker.check_positive_integer(dim)
    model.add(Linear(output_dim=dim))
    # model.add(BatchNorm())
    model.add(Activation('relu'))

  # Add lstm cells
  for dim in th.rc_dims:
    model.add(BasicLSTMCell(state_size=dim))

  # Add output layer
  # model.add(Linear(output_dim=th.output_dim))

  # Build model
  optimizer = tf.train.AdamOptimizer(th.learning_rate)
  model.build(optimizer)

  return model
