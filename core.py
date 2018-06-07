import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe.utils.misc import mark_str
from tframe.data.sequences.signals.signal import Signal
from tframe import console, SaveMode
from tframe.trainers import SmartTrainerHub
from tframe import Classifier

import data_utils as du
from gpat import init_methods
from gpat.gpat_bigdata import GPATBigData
from gpat.gpat_signal_set import GPATSignalSet


from_root = lambda path: os.path.join(ROOT, path)

th = SmartTrainerHub(as_global=True)
th.data_dir = from_root('99-GPAT/data')
from_gpat = lambda path: os.path.join(from_root('99-GPAT'), path)

# region : Paths

data_root = th.data_dir
from_data_root = lambda path: os.path.join(data_root, path)
input_dir = from_data_root('input')
output_dir = from_data_root('output')
from_input = lambda path: os.path.join(input_dir, path)
from_output = lambda path: os.path.join(output_dir, path)

label_sheet_path = from_data_root('labels.csv')
raw_train_data_path = from_input('raw_data/audio_train')
raw_train_csv_path = from_input('train.csv')
v_train_csv_path = from_input('audio_train_verified.csv')
raw_test_data_path = from_input('audio_test')

tfr_v_8000 = from_input('tfd_train_verified_8000Hz')
gpat_ss8000_v50 = from_input(
  'gpat_ss8000_v50.{}'.format(GPATSignalSet.EXTENSION))

# endregion : Paths

# th.input_shape = [200]
th.output_dim = 41
th.shuffle = False

th.allow_growth = False
th.gpu_memory_fraction = 0.4

th.save_mode = SaveMode.ON_RECORD
th.epoch_as_step = True

train_size = 20
val_size = 2
batches_per_epoch = 100


def activate():
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  input_dim = th.input_shape[0]
  init_f = init_methods.brutal_chop(input_dim)
  round_len_f = init_methods.brutal_chop_len_f(input_dim)

  # Load data
  train_set, val_set = du.load_balanced_data(
    tfr_v_8000, train_size, val_size, init_f=init_f, round_len_f=round_len_f)
  train_set.initialize(input_dim, batches_per_epoch)

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
    # x
    # model.train(train_set, validation_set=train_set, trainer_hub=th)
    # model.train(val_set, validation_set=val_set, trainer_hub=th)
  else:
    model.evaluate_model(train_set)
    model.evaluate_model(val_set)

  # End
  console.end()


if __name__ == '__main__':
  big_data = GPATBigData.load(tfr_v_8000)
  assert isinstance(big_data, GPATBigData)
  long_50 = big_data.pop_subset(50, length_prone='long', name='8000Hz_50L')
  assert isinstance(long_50, GPATBigData)
  ss = long_50.merge_to_signal_set(save_as=gpat_ss8000_v50)
  assert isinstance(ss, GPATSignalSet)
  # g_lens = [[len(ss.signals[i]) for i in indices] for indices in ss.groups]

  a = 1


  # bd = GPATData.load(tfr_v_8000)
  # ss = bd.load_data_set()
  # s = ss.signals[0]
  # assert isinstance(s, Signal)
  # s.plot(show_time_domain=True)

