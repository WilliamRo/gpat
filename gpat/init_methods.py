import numpy as np

import tframe.utils.misc as misc
from tframe import checker
from tframe import pedia
from tframe.data.sequences.signals.signal import Signal
from tframe.data.sequences.signals.signal_set import SignalSet
from tframe.data.bigdata import BigData


# region : Brutal Chop

def _brutal_chop(signal_set, size):
  assert isinstance(signal_set, SignalSet) and signal_set.size == 1
  checker.check_positive_integer(size)

  features, one_hots, dense_labels = [], [], []
  labels = signal_set.data_dict.get(pedia.labels, None)
  for i, s in enumerate(signal_set.signals):
    assert isinstance(s, Signal)
    if len(s) < size:
      s = np.pad(s, (0, size - len(s)), mode='constant', constant_values=0)
    num_steps = len(s) // size
    features.append(np.reshape(s[:num_steps * size], (num_steps, size)))
    if labels is not None:
      one_hots.append(np.tile(labels[i], (num_steps, 1)))
      dense_label = misc.convert_to_dense_labels(labels[i])
      dense_labels.append(dense_label)

  # Set features and targets to signal set
  signal_set.features = features
  if labels is not None:
    signal_set.targets = one_hots
    signal_set.data_dict['dense_labels'] = dense_labels

def brutal_chop(size):
  return lambda data: _brutal_chop(data, size)

def _brutal_chop_len_f(self, bs, ns, sz):
  assert isinstance(self, BigData)
  round_len = 0
  assert ns is not None
  for len_list in self.structure:
    checker.check_type(len_list, int)
    # For RNN models
    if ns < 0: round_len += len(len_list)
    else: round_len += int(sum([np.ceil(size // sz // bs / ns)
                                for size in len_list]))
  # Return round length
  return round_len

def brutal_chop_len_f(sz):
  return lambda self, bs, ns: _brutal_chop_len_f(self, bs, ns, sz)

# endregion : Brutal Chop

# region :

# endregion :
