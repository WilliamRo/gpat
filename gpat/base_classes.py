import pandas as pd

from tframe.data.base_classes import TFRData


class GPATBase(TFRData):

  LABEL_INDEX = 'label_index'
  DATA_INFO = 'data_info'
  GROUPS = 'groups'
  WAV_NAMES = 'wav_names'
  NUM_CLASSES = 41

  # region : Properties

  @property
  def label_index(self):
    df = self.properties[self.LABEL_INDEX]
    assert isinstance(df, pd.DataFrame)
    return df

  @property
  def data_info(self):
    df = self.properties[self.DATA_INFO]
    assert isinstance(df, pd.DataFrame)
    return df

  # endregion : Properties


