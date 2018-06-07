import collections
import numpy as np

from tframe import checker
from tframe import pedia
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal import Signal
from tframe.data.sequences.signals.signal_set import SignalSet

from gpat.base_classes import GPATBase


class GPATSignalSet(SignalSet, GPATBase):
  EXTENSION = 'gpss'

  INPUT_SIZE = 'input_size'
  BATCHES_PER_EPOCH = 'batches_per_epoch'
  INIT_F = 'init_f'
  READY = 'ready'

  # region ; Properties

  @property
  def is_ready(self):
    if not self.properties.get(self.READY, False):
      raise ValueError('!! GPAT Signal Set is not ready yet. Initialize it'
                       ' before using.')
    return True

  @property
  def input_size(self):
    val = self.properties.get(self.INPUT_SIZE)
    checker.check_positive_integer(val)
    return val

  @property
  def batches_per_epoch(self):
    val = self.properties.get(self.BATCHES_PER_EPOCH)
    checker.check_positive_integer(val)
    return val

  @property
  def init_f(self):
    return self.properties.get(self.INIT_F, self.brutal_chop)

  @init_f.setter
  def init_f(self, val):
    assert callable(val)
    self.properties[self.INIT_F] = val

  @property
  def groups(self):
    val = self.properties[self.GROUPS]
    assert isinstance(val, list) and len(val) == self.NUM_CLASSES
    return val

  # endregion : Properties

  # region : Public Methods

  def initialize(self, input_size, batches_per_epoch):
    checker.check_positive_integer(input_size)
    checker.check_positive_integer(batches_per_epoch)

    self.properties[self.INPUT_SIZE] = input_size
    self.properties[self.BATCHES_PER_EPOCH] = batches_per_epoch
    self.properties[self.READY] = True

  def get_round_length(self, batch_size, num_steps=None):
    return None

  def gen_batches(self, batch_size, **kwargs):
    assert self.is_ready
    checker.check_positive_integer(self.batches_per_epoch)
    for i in range(self.batches_per_epoch):
      matrix, labels = self._random_signal_matrix(batch_size, self.input_size)
      batch = DataSet(matrix, labels)
      batch.name = 'gpat_{}of{}'.format(i + 1, self.batches_per_epoch)
      yield batch

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, *args, **kwargs):
    assert self.is_ready
    checker.check_positive_integer(self.batches_per_epoch)
    for i in range(self.batches_per_epoch):
      matrix, labels = self._random_signal_matrix(batch_size)
      matrix, labels = self.init_f(matrix, labels)
      for batch in self._gen_rnn_batches(matrix, labels, num_steps):
        assert isinstance(batch, DataSet)
        batch.name += '_in_{}of{}'.format(i + 1, self.batches_per_epoch)
        yield batch

  # endregion : Public Methods

  # region : Private Methods

  def _random_signal_matrix(self, batch_size=1, depth=None):
    # :: Step 1 - Randomly select signals
    classes = np.random.randint(self.NUM_CLASSES, size=batch_size)
    indices = []
    for cls in classes:
      group_index = np.random.randint(len(self.groups[cls]))
      indices.append(self.groups[cls][group_index])
    checker.check_type(indices, int)

    # :: Step 2 - Generate a regular numpy array
    #             of shape (batch_size, min(selected_signal_length))
    signal_lens = [len(self.signals[i]) for i in indices]
    checker.check_positive_integer(self.input_size)
    depth = max(min(signal_lens), self.input_size) if depth is None else depth
    matrix = np.zeros((batch_size, depth))
    labels = []
    for i in range(batch_size):
      s = self.signals[indices[i]]
      s_len = len(s)
      # If s is shorter than depth, pad it with 0
      if s_len < depth:
        head = np.random.randint(depth - s_len + 1)
        tail = depth - s_len - head
        s = np.pad(s, (head, tail), mode='constant', constant_values=0)
        s_len = len(s)
      assert isinstance(s, np.ndarray)

      offset = np.random.randint(s_len - depth + 1)
      matrix[i] = s[offset:offset + depth]
      labels.append(self.data_dict[pedia.labels][indices[i]])

    # Concatenate labels to be of shape (batch_size, 41)
    labels = np.concatenate(labels)
    # Return matrix and labels
    return matrix, labels

  def _gen_rnn_batches(self, x, y, num_steps, *args):
    # Sanity check
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert isinstance(num_steps, int)
    assert len(x.shape) == 3 and x.shape[2] == self.input_size
    steps = x.shape[1]
    assert y.shape == (x.shape[0], steps, self.NUM_CLASSES)

    # Yield RNN batches
    if num_steps < 0: num_steps = steps
    yield_times =  int(np.ceil(steps / num_steps))
    for i in range(yield_times):
      batch_x = x[:, i * num_steps:min((i + 1) * num_steps, steps)]
      batch_y = y[:, i * num_steps:min((i + 1) * num_steps, steps)]
      batch = DataSet(batch_x, batch_y, in_rnn_format=True)

      # State should be reset at the beginning of a sequence
      if i == 0: batch.should_reset_state = True
      batch.name = 'gpat_{}of{}'.format(i + 1, yield_times)
      yield batch

  # endregion : Private Methods

  # region : Init Methods

  def brutal_chop(self, matrix, labels):
    # Sanity check
    assert isinstance(matrix, np.ndarray) and len(matrix.shape) == 2
    assert isinstance(labels, np.ndarray)
    batch_size, depth = matrix.shape
    assert list(labels.shape) == [batch_size, self.NUM_CLASSES]

    # Chop matrix brutally
    steps = depth // self.input_size
    matrix_c = np.reshape(matrix[:, :steps * self.input_size],
                          (batch_size, steps, self.input_size))
    labels = np.reshape(labels, (batch_size, 1, self.NUM_CLASSES))
    labels_c = np.broadcast_to(labels, (batch_size, steps, self.NUM_CLASSES))

    # Return
    return matrix_c, labels_c


  # endregion : Init Methods


if __name__ == '__main__':
  import core
  # big_data = core.GPATBigData.load(core.tfr_v_8000)
  # assert isinstance(big_data, core.GPATBigData)
  # long_50 = big_data.pop_subset(50, length_prone='long', name='8000Hz_50L')
  # assert isinstance(long_50, core.GPATBigData)
  # ss = long_50.merge_to_signal_set(save_as=core.gpat_ss8000_v50)
  # assert isinstance(ss, GPATSignalSet)
  gpss = GPATSignalSet.load(core.gpat_ss8000_v50)
  # assert isinstance(gpss, GPATSignalSet)
  gpss.initialize(2000, 10)
  for data in gpss.gen_rnn_batches(8, 4):
    name = data.name
    a = 1

