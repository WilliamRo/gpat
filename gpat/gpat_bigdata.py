import collections
import copy
import numpy as np
import os
import pandas as pd

from tframe import console
from tframe import checker
from tframe import pedia
from tframe.data.bigdata import BigData
from tframe.data.sequences.signals.signal_set import SignalSet

from gpat import raw_data_agent as du
from gpat.base_classes import GPATBase
from gpat.gpat_signal_set import GPATSignalSet


class GPATBigData(BigData, GPATBase):
  """"""
  FILE_NAME = 'gpat.meta'

  def __init__(self, data_dir, csv_path=None, lb_sheet_path=None):
    # Call parent's constructor
    super().__init__(data_dir, save=False)
    # Initialize big data
    self.with_labels = False
    self._init_big_data(csv_path, lb_sheet_path)
    self.save()

  # region : Properties

  @property
  def min_length(self):
    return min(np.concatenate(list(self.files.values())))

  @property
  def max_length(self):
    return max(np.concatenate(list(self.files.values())))

  @property
  def groups(self):
    val = self.properties[self.GROUPS]
    assert isinstance(val, collections.OrderedDict)
    return val

  @property
  def group_population(self):
    population = collections.OrderedDict()
    for label, files in self.groups.items():
      population[label] = len(files)
    return population

  @property
  def group_size(self):
    size = collections.OrderedDict()
    for label, files in self.groups.items():
      size[label] = sum(list(map(self._get_length, files)))
    return size
  
  @property
  def smallest_population(self):
    return min(self.group_population.values())

  # endregion : Properties

  # region : Private Methods

  def _init_big_data(self, csv_path, lb_sheet_path):
    if csv_path is None or lb_sheet_path is None: return
    self.with_labels = True
    self.properties[self.DATA_INFO] = pd.read_csv(csv_path)
    self.properties[self.LABEL_INDEX] = pd.read_csv(lb_sheet_path)
    # Generate groups
    self.properties[self.GROUPS] = collections.OrderedDict()
    console.show_status('Generating group information ...')
    for i, label in enumerate(self.label_index[du.LABEL]):
      file_list = list(
        self.data_info.loc[self.data_info[du.LABEL] == label][du.FNAME])
      self.groups[label] = sorted(file_list, key=self._get_length, reverse=True)
      console.print_progress(i, len(self.label_index))
    console.show_status('Group population:')
    console.pprint(self.group_population)

  def _get_length(self, file_name):
    length = [val[0] for key, val in self.files.items() if file_name in key]
    assert len(length)== 1
    return length[0]

  def _copy_empty_container(self, name='subset'):
    data_set = copy.deepcopy(self)
    data_set.files = {}
    data_set.properties[self.GROUPS] = collections.OrderedDict()
    data_set.name = name
    return data_set

  # endregion : Private Methods

  # region : Public Methods

  def merge_to_signal_set(self, save_as=None):
    signals = []
    onehot_labels = []
    file_names = []
    groups = []

    # Put each signal together with its one-hot label into lists
    for wav_names in self.groups.values():
      indices = []
      for wav_name in wav_names:
        file_path = os.path.join(self.data_dir, wav_name + '.tfds')
        # Load signal set from disk
        data_set = self._load_data_set(file_path)
        assert isinstance(data_set, SignalSet)
        assert len(data_set.signals) == 1 and len(data_set[pedia.labels]) == 1
        # Append signal and label to corresponding list
        signals.append(data_set.signals[0])
        onehot_labels.append(data_set[pedia.labels][0])
        file_names.append(wav_name)
        # Print progress
        console.print_progress(len(signals), len(self.files))
        indices.append(len(signals) - 1)

      groups.append(indices)

    # Wrap data_dict and properties
    data_dict, properties = {}, {}
    data_dict[pedia.labels] = onehot_labels
    properties[self.LABEL_INDEX] = self.label_index
    properties[self.DATA_INFO] = self.data_info
    properties[self.WAV_NAMES] = file_names
    properties[self.GROUPS] = groups

    # Save and return
    gpat_signal_set = GPATSignalSet(signals, data_dict=data_dict, **properties)
    if save_as is not None: gpat_signal_set.save(save_as)
    return gpat_signal_set

  def pop_subset(self, size, length_prone='random', name='subset'):
    """Pop sub data set containing 'size' samples for each class"""
    # Check input parameters
    checker.check_positive_integer(size)
    assert length_prone in ('short', 'long', 'random')
    if size > self.smallest_population:
      raise ValueError('!! size should be less than the smallest population '
                       'of this data set: {}'.format(self.smallest_population))
    # Check classes
    assert len(self.groups) == self.NUM_CLASSES

    # Fill subset
    subset = self._copy_empty_container()
    subset.name = name
    for label, wavs in self.groups.items():
      assert isinstance(wavs, list)
      # Initialize file list for label class
      subset.groups[label] = []
      # Pop data from self
      for _ in range(size):
        if length_prone == 'short': i = -1
        elif length_prone == 'long': i = 0
        else: i = np.random.randint(len(wavs))
        # Pop from self
        file_name = wavs.pop(i)
        subset.groups[label].append(file_name)
        tfd_file_name = file_name + '.tfds'
        subset.files[tfd_file_name] = self.files.pop(tfd_file_name)

    # Check length
    assert len(self.files) == sum(list(self.group_population.values()))
    assert len(subset.files) == sum(list(subset.group_population.values()))
    return subset

  # endregion : Public Methods


if __name__ == '__main__':
  import core
  tfr_v_8000 = core.from_input('tfd_train_verified_8000Hz')
  v_csv = core.from_input('audio_train_verified.csv')
  uv_csv = core.from_input('audio_train_unverified.csv')

  data = GPATBigData(tfr_v_8000, v_csv, core.label_sheet_path)
  # data = GPATData.load(tfr_v_8000)
  assert isinstance(data, GPATBigData)
  console.show_status(data.size)

  subset = data.pop_subset(5)
  console.show_status(data.size)
  console.show_status(subset.size)

