import os
import numpy as np
import librosa
import pandas as pd

from tframe import console
from tframe import pedia
from tframe.utils.local import check_path
import tframe.utils.misc as misc

from tframe.data.sequences.signals.signal import Signal
from tframe.data.sequences.signals.signal_set import SignalSet
from tframe.data.bigdata import BigData

from gpat import raw_data_agent as du


def wav_to_signal(file_path, fs):
  # Sanity check
  assert os.path.isfile(file_path) and file_path[-4:] == '.wav'
  wav, fs = librosa.core.load(file_path, sr=fs, res_type='kaiser_fast')
  return Signal(wav, fs)


def convert_to_tframe_files(
    data_dir, fs, csv_path=None, label_path=None, to_dir=None, verbose=True):
  # Check paths
  check_path(data_dir, create_path=False)
  if csv_path is not None: check_path(csv_path, create_path=False)
  if label_path is not None: check_path(label_path, create_path=False)
  if to_dir is None: to_dir = os.path.join(
      os.path.dirname(data_dir), 'tfd_{}_{}Hz'.format(
      os.path.basename(data_dir), fs))
  check_path(to_dir, create_path=True)

  # Generate file list
  srcs, dsts = [], []
  for f in os.listdir(data_dir):
    file_path = os.path.join(data_dir, f)
    if not os.path.isfile(file_path) or f[-4:] != '.wav': continue
    srcs.append(file_path)
    dsts.append(os.path.join(to_dir, f))

  # Load csv file and label sheet if necessary
  csv, lb_sheet = None, None
  if csv_path is not None and label_path is not None:
    csv = pd.read_csv(csv_path)
    lb_sheet = pd.read_csv(label_path)

  # Wrap each .wav file in data_dir into a SignalSet
  num_files = len(srcs)
  console.show_status('Converting {} ...'.format(data_dir))
  for i, src, dst in zip(range(num_files), srcs, dsts):
    s = wav_to_signal(src, fs)
    ss = SignalSet(s, name=os.path.basename(src))
    if csv is not None:
      fname = os.path.basename(src)
      ss.data_dict[pedia.labels] = [_get_one_hot(fname, csv, lb_sheet)]

    ss.save(dst)
    if verbose: console.print_progress(i + 1, num_files)

  # Show status
  console.show_status('Data saved to {}'.format(to_dir))


def init_big_data(data_dir, csv_path=None, lb_sheet_path=None):
  # Initiate a big data
  bd = BigData.load(data_dir)
  #
  if csv_path is not None and lb_sheet_path is not None:
    csv = pd.read_csv(csv_path)
    lb_sheet = pd.read_csv(lb_sheet_path)

  # Save bid data meta file
  bd.save()


def _get_one_hot(fname, csv, lb_sheet):
  assert isinstance(csv, pd.DataFrame) and isinstance(lb_sheet, pd.DataFrame)
  labels = csv.loc[csv[du.FNAME] == fname][du.LABEL]
  assert len(labels) == 1
  label = list(labels)[0]
  index = list(lb_sheet.loc[lb_sheet[du.LABEL] == label]['index'])[0]
  return misc.convert_to_one_hot([index], 41)


if __name__ == '__main__':
  fs = 8000
  name = 'audio_train_verified'
  data_dir = '../data/input/{}'.format(name)
  csv_path = '../data/input/{}.csv'.format(name)
  label_sheet_path = '../data/labels.csv'
  to_dir = '../data/input/tfd_train_verified_{}Hz'.format(fs)
  bd = BigData(to_dir)

  # bd = BigData.load(to_dir)
  a = 1
  # for batch in bd.gen_batches(batch_size=5400):
  #   a = 1

  # convert_to_tframe_files(data_dir, fs, csv_path, label_sheet_path, to_dir)

