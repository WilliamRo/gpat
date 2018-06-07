import os
import collections
import pandas as pd
import shutil as sh
import librosa

from tframe import console
from tframe.utils.local import check_path


DATA_DIR = '../data'
LABEL_CSV_PATH = os.path.join(DATA_DIR, 'labels.csv')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
RAW_TRAIN_DIR = os.path.join(INPUT_DIR, 'raw_data', 'audio_train')
RAW_TRAIN_CSV_PATH = os.path.join(INPUT_DIR, 'train.csv')

FNAME = 'fname'
LABEL = 'label'
MANUALLY_VERIFIED = 'manually_verified'


def generate_label_sheet(csv_path):
  # Check path
  check_path(csv_path, create_path=False)
  check_path(DATA_DIR, create_path=True)

  # Read raw csv
  raw_csv = pd.read_csv(csv_path)
  label_descriptions = list(raw_csv[LABEL].unique())
  labels = collections.OrderedDict()
  labels[LABEL] = label_descriptions
  labels['index'] = list(range(len(label_descriptions)))
  label_sheet = pd.DataFrame(labels)

  # Write label sheet
  label_sheet.to_csv(LABEL_CSV_PATH, index=False)
  console.show_status('Label sheet saved to {}'.format(LABEL_CSV_PATH))


def separate_verified_data(data_dir, csv_path, to_path, verbose=True,
                           vname='audio_train_verified',
                           uvname='audio_train_unverified'):
  # Check path
  check_path(data_dir, create_path=False)
  check_path(csv_path, create_path=False)

  # Read raw csv file and split
  raw_csv = pd.read_csv(csv_path)
  verified_csv = raw_csv.loc[raw_csv[MANUALLY_VERIFIED] == 1]
  unverified_csv = raw_csv.loc[raw_csv[MANUALLY_VERIFIED] == 0]

  # Copy verified data
  for df, dn in ((verified_csv, vname), (unverified_csv, uvname)):
    check_path(to_path, dn, create_path=True)
    console.show_status('Generating {} data ... '.format(dn))
    num_files = len(df)
    for i, file_name in enumerate(df[FNAME]):
      from_file = os.path.join(data_dir, file_name)
      to_file = os.path.join(to_path, dn, file_name)
      sh.copyfile(from_file, to_file)
      if verbose: console.print_progress(i, num_files)
    file_name = to_path + dn + '.csv'
    df.to_csv(file_name, index=False)
    console.show_status('CSV file saved to {}'.format(file_name))


def down_sample(data_dir, sample_rate, to_path=None, verbose=True):
  """ Common sample frequency list:
   8000 Hz - fs for telephone
  11025 Hz -
  22050 Hz - fs for radio
  32000 Hz - fs for miniDV
  44100 Hz - fs for CD
  """
  # Check data directory
  check_path(data_dir, create_path=False)

  # Check to_path
  if to_path is None: to_path = os.path.join(
    os.path.dirname(data_dir), '{}_{}Hz'.format(
      os.path.basename(data_dir), sample_rate))
  check_path(to_path, create_path=True)

  # Generate file list
  srcs, dsts = [], []
  for f in os.listdir(data_dir):
    file_path = os.path.join(data_dir, f)
    if not os.path.isfile(file_path) or f[-4:] != '.wav': continue
    srcs.append(file_path)
    dsts.append(os.path.join(to_path, f))

  # Down sample each .wav file in data_dir
  num_files = len(srcs)
  console.show_status('Down sampling ...')
  for i, src, dst in zip(range(num_files), srcs, dsts):
    data, _ = librosa.core.load(src, sample_rate, res_type='kaiser_fast')
    librosa.output.write_wav(dst, data, sample_rate)
    if verbose: console.print_progress(i + 1, num_files)

  # Show status
  console.show_status('Data saved to {}'.format(to_path))


if __name__ == '__main__':
  separate_verified_data(RAW_TRAIN_DIR, RAW_TRAIN_CSV_PATH, INPUT_DIR)
  # generate_label_sheet(RAW_TRAIN_CSV_PATH)
  # down_sample(RAW_TRAIN_DIR, 8000, to_path=INPUT_DIR)



