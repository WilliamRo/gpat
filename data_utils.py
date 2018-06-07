from tframe import console

import core
from gpat.gpat_bigdata import GPATBigData
from gpat.gpat_signal_set import GPATSignalSet


def load_data_v1():
  pass

def load_balanced_data(data_dir, train_size=20, validation_size=2,
                       init_f=None, round_len_f=None):
  # Load data from data_dir
  data_set = GPATBigData.load(data_dir, csv_path=core.v_train_csv_path,
                              lb_sheet_path=core.label_sheet_path)
  assert isinstance(data_set, GPATBigData)
  assert data_set.with_labels

  # Pop subsets from data_set
  train_set = data_set.pop_subset(
    train_size, length_prone='long', name='train_set')
  val_set = data_set.pop_subset(
    validation_size, length_prone='short', name='validation_set')
  train_set.init_f = init_f
  train_set.round_len_f = round_len_f
  val_set.init_f = init_f
  val_set.round_len_f = round_len_f

  # Show status
  console.show_status('Train set (size={})'.format(train_set.size))
  console.supplement('min_len = {}'.format(train_set.min_length))
  console.supplement('max_len = {}'.format(train_set.max_length))
  console.show_status('Validation set (size={})'.format(val_set.size))
  console.supplement('min_len = {}'.format(val_set.min_length))
  console.supplement('max_len = {}'.format(val_set.max_length))

  return train_set.merge_to_signal_set(), val_set


if __name__ == '__main__':
  import gpat.init_methods as init_methods
  train_set, val_set = load_balanced_data(
    core.tfr_v_8000, init_f=init_methods.brutal_chop(200))
  # for data in train_set.gen_rnn_batches(num_steps=100):
  #   a = 1
